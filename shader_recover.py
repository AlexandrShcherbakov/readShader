"""
The only goal of this module is to restore HLSL code by DXIL disasm.
Use recover method. It returns restored HLSL.
"""

import collections
import copy
import typing



class VarAccessor:
    """
    This is a temp class which generates a string for viriables access.
    TODO: Remove this class when variable accessor will be a type as _Constant
    """
    def __init__(self, tp, comps):
        self.single_var = True
        self.name = comps[0][0]
        self.components = "".join(comp[1] for comp in comps)
        if any(x[0] != self.name for x in comps):
            self.name = f"{tp}{len(comps)}({', '.join(map(lambda x: f'{x[0]}.{x[1]}', comps))})"
            self.components = ""
            self.single_var = False

    def __str__(self):
        if self.single_var:
            return self.name + "." + self.components
        return self.name


class _AssignStatement:
    SubstituteParams = collections.namedtuple(
        "SubstituteParams",
        ["hlsl_instruction", "has_brackets"]
    )

    instructions = {
        "utof": SubstituteParams("asfloat({})", True),
        "itof": SubstituteParams("asfloat({})", True),
        "mad": SubstituteParams("{} * {} + {}", False),
        "ftou": SubstituteParams("asuint({})", True),
        "mov": SubstituteParams("{}", True),
        "ld_indexable(texture2darray)(float,float,float,float)":
            SubstituteParams("{1.name}[{0}].{1.components}", True),
        "ftoi": SubstituteParams("asint({})", True),
        "ine": SubstituteParams("{} != {}", False),
        "ne": SubstituteParams("{} != {}", False),
        "eq": SubstituteParams("{} == {}", False),
        "ieq": SubstituteParams("{} == {}", False),
        "lt": SubstituteParams("{} < {}", False),
        "mul": SubstituteParams("{} * {}", True),
        "ge": SubstituteParams("{} >= {}", False),
        "movc": SubstituteParams("{} ? {} : {}", False),
        "div": SubstituteParams("{} / {}", True),
        "frc": SubstituteParams("frac({})", True),
        "ld_structured_indexable(structured_buffer,stride=16)(mixed,mixed,mixed,mixed)":
            SubstituteParams("{2.name}[{0} + {1}].{2.components}", True),
        "and": SubstituteParams("{} & {}", False),
        "add": SubstituteParams("{} + {}", False),
        "dp3": SubstituteParams("dot({}, {})", True),
        "dp2": SubstituteParams("dot({}, {})", True),
        "dp4": SubstituteParams("dot({}, {})", True),
        "sqrt": SubstituteParams("sqrt({})", True),
        "rsq": SubstituteParams("rsqrt({})", True),
        "round_ni": SubstituteParams("floor({})", True),
        "round_z": SubstituteParams("trunc({})", True),
        "exp": SubstituteParams("exp({})", True),
        "min": SubstituteParams("min({}, {})", True),
        "max": SubstituteParams("max({}, {})", True),
        "mul_sat": SubstituteParams("saturate({} * {})", True),
        "div_sat": SubstituteParams("saturate({} / {})", True),
        "ld_indexable": SubstituteParams("{1.name}[{0}].{1.components}", True),
        "ld_structured": SubstituteParams("{2.name}[{0} + {1}].{2.components}", True),
        "sample_l": SubstituteParams("{1.name}.SampleLevel({2}, {0}, {3}).{1.components}", True),
        "sample_indexable(texture2d)(float,float,float,float)":
            SubstituteParams("{1.name}.Sample({2}, {0}).{1.components}", True),
        "mad_sat": SubstituteParams("saturate({} * {} + {})", True),
        "add_sat": SubstituteParams("saturate({} + {})", True),
        "log": SubstituteParams("log({})", True),
        "sample_l(texturecube)(float,float,float,float)":
            SubstituteParams("{1.name}.SampleLevel({2}, {0}, {3}).{1.components}", True),
        "mov_sat": SubstituteParams("saturate({})", True),
    }

    def __init__(self, raw_tokens: list[str]) -> None:
        self.instruction = raw_tokens[0]
        self.result = raw_tokens[1]
        self.operands = raw_tokens[2:]
        self.initializer = False
        self.type_id = -1

    def get_computation(self):
        """Get string of expression which we store in a variable"""
        type_name = type_idx[self.type_id].__name__
        proccessed_operands = [
            VarAccessor(type_name, x) if isinstance(x, list) else x
            for x in self.operands
        ]

        return self.instructions[self.instruction][0].format(*proccessed_operands)

    def __repr__(self):
        type_comps = str(len(self.result)) if len(self.result) > 1 else ""
        type_name = type_idx[self.type_id].__name__
        prefix = (type_name + type_comps + " ") if self.initializer else ""
        result_name = self.result[0][0]
        for res, _ in self.result:
            if res != result_name:
                raise Exception(f"Result names don't match! {result_name} and {res}")

        return f"{prefix}{result_name} = {self.get_computation()};"


class _FlowControlStatement:
    SubstituteParams = collections.namedtuple(
        "SubstituteParams",
        ["hlsl_instruction", "depth_corrector"]
    )

    instructions = {
        "if_nz": SubstituteParams("if ({}) {{", 0),
        "else": SubstituteParams("}} else {{", -1),
        "endif": SubstituteParams("}}", -1),
        "ret": SubstituteParams("return;", 0),
        "discard_nz": SubstituteParams("if ({}) discard;", 0),
    }
    def __init__(self, raw_tokens: list[str]) -> None:
        self.instruction = raw_tokens[0]
        self.operands = raw_tokens[1:]

    def __repr__(self):
        proccessed_operands = [
            VarAccessor("bool", x) if isinstance(x, list) else x
            for x in self.operands
        ]

        return self.instructions[self.instruction].hlsl_instruction.format(*proccessed_operands)

    def get_depth(self):
        """Returns depth corrector for this instruction (used for else, endif)"""
        return self.instructions[self.instruction].depth_corrector


class _ModifierStatement:
    SubstituteParams = collections.namedtuple("SubstituteParams", ["hlsl_instruction"])

    instructions = {
        "atomic_iadd": SubstituteParams("InterlockedAdd({0.name}[{1}], {2})"),
        "store_structured": SubstituteParams("{0.name}[{1} + {2}].{0.components} = {3}"),
        "resinfo_indexable(texturecube)(float,float,float,float)_float":
            SubstituteParams("{2.name}.GetDimensions({1}, {0})"),
        "sincos": SubstituteParams("sincos({}, {}, {})"),
    }
    def __init__(self, raw_tokens: list[str]) -> None:
        self.instruction = raw_tokens[0]
        self.operands = raw_tokens[1:]

    def __repr__(self):
        proccessed_operands = [
            VarAccessor("bool", x) if isinstance(x, list) else x
            for x in self.operands
        ]
        hlsl_inst = self.instructions[self.instruction].hlsl_instruction
        return hlsl_inst.format(*proccessed_operands) + ";"



class _InitStatement:
    SubstituteParams = collections.namedtuple("SubstituteParams", ["hlsl_instruction"])

    instructions = {
        "init": SubstituteParams("{0}")
    }
    def __init__(self, var_name, tp, size) -> None:
        self.result = var_name
        self.type_id = tp
        self.size = size

    def __repr__(self):
        type_comps = str(self.size) if self.size > 1 else ""
        return type_idx[self.type_id].__name__ + type_comps + " " + self.result.name + ";"


_Statement = typing.Union[_AssignStatement, _FlowControlStatement, _ModifierStatement]


_STATEMENT_CLASSIFIER:dict[str, _Statement] = dict()
for statement_type in [_AssignStatement, _FlowControlStatement, _ModifierStatement]:
    for instruction in statement_type.instructions:
        _STATEMENT_CLASSIFIER[instruction] = statement_type


class _CodeBlock:
    def __init__(self, statements:list[_Statement], depth:int) -> None:
        self.statements = statements
        self.depth = depth

    def __repr__(self):
        return "\n".join(list(map(lambda x: "  " * self.depth + str(x), self.statements)))


def _filter_disasm_statements(disasm: str) -> tuple[list[str], list[str]]:
    header = []
    statements = []
    for line in disasm.split("\n"):
        if ":" in line:
            statements.append(line.split(": ", 1)[1])
        elif line.strip().startswith("dcl_"):
            header.append(line.strip())
    return header, statements


def _statement_str_to_tokens(raw_line: str) -> list[str]:
    def squash_brackets(tokens, begin, end, sep):
        i = len(tokens) - 1
        while i >= 0:
            while begin in tokens[i] and end not in tokens[i]:
                tokens[i] += sep + tokens[i + 1]
                del tokens[i + 1]
            i -= 1

    tokens = list(filter(lambda x: len(x) > 0, map(lambda x: x.strip(', '), raw_line.split(' '))))
    squash_brackets(tokens, "(", ")", ",")
    squash_brackets(tokens, "[", "]", "")
    return tokens


def _split_commands_on_blocks(
    statements: list[_Statement]
) -> tuple[list[_CodeBlock], list[list[int]]]:
    current_depth = 0
    blocks = [_CodeBlock([], current_depth)]
    flow_controllers = { "if_nz": 1, "else": 0, "endif": -1 }
    previous_block_links : list[list[int]] = [[]] 
    for command in statements:
        blocks[-1].statements.append(command)
        if command.instruction not in flow_controllers:
            continue
        current_depth += flow_controllers[command.instruction]
        if blocks[-1].depth == current_depth - 1:
            previous_blocks = [len(blocks) - 1]
        elif blocks[-1].depth == current_depth:
            for idx in range(len(blocks) - 2, -1, -1):
                if blocks[idx].depth == current_depth - 1:
                    previous_blocks = [idx]
                    break
        elif blocks[-1].depth == current_depth + 1:
            previous_blocks = []
            for idx in range(len(blocks) - 2, -1, -1):
                if (
                    blocks[idx].depth == blocks[-1].depth
                    and blocks[idx + 1].depth == blocks[idx].depth
                ):
                    previous_blocks.append(idx)
                    break
            previous_blocks.append(len(blocks) - 1)
        blocks.append(_CodeBlock([], current_depth))
        previous_block_links.append(previous_blocks)
    return blocks, previous_block_links


def _build_statements(raw_tokens_collection: list[list[str]]) -> list[_Statement]:
    result = []
    for raw_tokens in raw_tokens_collection:
        result.append(_STATEMENT_CLASSIFIER[raw_tokens[0]](raw_tokens))
    return result


class _GraphNode:
    def __init__(self, next_blocks:list[int], prev_blocks:list[int]) -> None:
        self.next_blocks = next_blocks
        self.prev_blocks = prev_blocks

    def __repr__(self):
        return f"-> {self.next_blocks}\n <- {self.prev_blocks}"


def _gen_graph(prev_block_links:list[list[int]]) -> list[_GraphNode]:
    result = []
    next_blocks:dict[int, list[int]] = dict()
    for block_id in range(len(prev_block_links) - 1, -1, -1):
        next_ids = next_blocks.get(block_id, [])
        result.append(_GraphNode(next_ids, prev_block_links[block_id]))
        for prev_block in prev_block_links[block_id]:
            if prev_block not in next_blocks:
                next_blocks[prev_block] = []
            next_blocks[prev_block].append(block_id)
    return result[::-1]


def _is_register(operand:str) -> bool:
    return (
        isinstance(operand, str)
        and operand.startswith("r")
        and operand[1:].split(".")[0].isdigit()
    )


def _operand_uses_register(operand, register, components):
    operand = operand.strip("-")
    return (
        _is_register(operand)
        and operand.split(".")[0] == register
        and any(comp in components for comp in operand.split(".")[1])
    )

type_idx = [None, bool, int, float]


def _get_token_type(token, operand_to_type):
    if isinstance(token, str):
        return _get_type_of_raw_token(token, operand_to_type)
    elif isinstance(token, list):
        return type_idx[max(type_idx.index(_get_token_type(t, operand_to_type)) for t in token)]
    return token.get_type()


def _get_type_of_raw_token(token, operand_to_type):
    if _is_register(token):
        return operand_to_type[token.split(".")[0]]
    return float if "." in token else int


class _Variable:
    def __init__(self, name, register, components, usages, init_pos, mod_pos):
        self.name = name
        self.register = register
        self.components = components
        self.usages = usages
        self.init_pos = init_pos
        self.modifiers = {copy.copy(mod_pos): copy.copy(components)}
        self.type_id = -1

    def get_var_comp(self, component):
        """Returns variable component by register component"""
        return "xyzw"[list(sorted(list(self.components))).index(component)]


class _VariablesContext:
    def __init__(self, input_constants, header, var_names):
        self.variables = dict()
        self.vars_count = 0
        self.input_constants = input_constants
        self.input_resources = _get_types_for_resources(header)
        self.var_names = var_names

    def add_var(self, usages, register, components, init_pos, mod_pos):
        """Adds variable to context"""
        var_name = f"var_{self.vars_count}"
        if var_name in self.var_names:
            var_name = self.var_names[var_name]
        self.vars_count += 1
        self.variables[var_name] = _Variable(
            var_name, register, components, usages, init_pos, mod_pos
        )

    def extend_var(self, var_name, usages, components, graph, modifier_pos):
        """Extends variable usages list by merging"""
        self.variables[var_name].modifiers[modifier_pos] = components
        for block_id, block_us in usages.items():
            if block_id not in self.variables[var_name].usages:
                self.variables[var_name].usages[block_id] = block_us
                continue
            for statement_id, statement_us in block_us.items():
                statements = self.variables[var_name].usages[block_id]
                if statement_id not in statements:
                    statements[statement_id] = statement_us
                    continue
                for operand_id, operand_us in statement_us.items():
                    if operand_id not in statements[statement_id]:
                        statements[statement_id][operand_id] = operand_us
                        continue
                    statements[statement_id][operand_id] |= operand_us
        if all(comp in self.variables[var_name].components for comp in components):
            return
        self.variables[var_name].components |= components
        block_id = (
            self.variables[var_name].init_pos[0]
            if isinstance(tuple, self.variables[var_name].init_pos)
            else self.variables[var_name].init_pos
        )
        self.variables[var_name].init_pos = (
            _find_parent_block(graph, self.variables[var_name].usages)
        )


class _Constant:
    def __init__(self, values, comp_len, tp):
        self.values = values
        self.comp_len = comp_len
        self.type = tp

    @staticmethod
    def try_to_parse(token, context):
        """Returns None if can't parse token with current type. Otherwise creates class instance"""
        if not token.startswith("l("):
            return None
        values = ", ".join(token[2:-1].split(","))
        comp_len = 4 if "," in values else 1
        tp = float if "." in values else int
        return _Constant(values, comp_len, tp)

    def __str__(self):
        if self.comp_len == 1:
            return self.values
        return f"{self.type.__name__}{self.comp_len}({self.values})"

class _BuiltinConstant:
    def __init__(self, name, components, tp):
        self.name = name
        self.components = components
        self.type = tp

    @staticmethod
    def try_to_parse(token, context):
        """Returns None if can't parse token with current type. Otherwise creates class instance"""
        built_in_consts = {
            "vThreadID": int
        }
        if "." not in token:
            return None

        name, components = token.split(".")
        if name not in built_in_consts:
            return None

        return _BuiltinConstant(name, components, built_in_consts[name])

    def __str__(self):
        return f"{self.name}.{self.components}"


class _InputConstant:
    def __init__(self, name, components, tp, decorator):
        self.name = name
        self.components = components
        self.type = tp
        self.decorator = decorator

    @staticmethod
    def try_to_parse(token, context):
        """Returns None if can't parse token with current type. Otherwise creates class instance"""
        decorator = "{}"
        if token[0] == "-":
            decorator = "-{}"
            token = token[1:]
        if "." not in token:
            return None

        name, components = token.split(".")
        if name not in context.input_constants:
            return None

        return _InputConstant(name, components, context.input_constants[name], decorator)

    def __str__(self):
        return self.decorator.format(f"{self.name}.{self.components}")

class _InputResource:
    def __init__(self, name, components, tp, decorator):
        self.name = name
        self.components = components
        self.type = tp
        self.decorator = decorator

    @staticmethod
    def try_to_parse(token, context):
        """Returns None if can't parse token with current type. Otherwise creates class instance"""
        decorator = "{}"
        if token[0] == "-":
            decorator = "-{}"
            token = token[1:]
        if "." not in token:
            return None

        name, components = token.split(".")
        if name not in context.input_resources:
            return None

        return _InputResource(name, components, context.input_resources[name], decorator)

    def __str__(self):
        return self.decorator.format(f"{self.name}.{self.components}")

class _OutputResource:
    def __init__(self, name, tp):
        self.name = name
        self.type = tp

    @staticmethod
    def try_to_parse(token, context):
        """Returns None if can't parse token with current type. Otherwise creates class instance"""
        if token not in context.input_resources:
            return None

        return _OutputResource(token, context.input_resources[token])

    def __str__(self):
        return self.name


def _substitute_operands(statement, pos, context):
    for operand_id, operand in enumerate(statement.operands):
        types_to_process = [
            _Constant, _BuiltinConstant, _InputConstant, _InputResource, _OutputResource
        ]
        for token_type in types_to_process:
            if processed := token_type.try_to_parse(operand, context):
                statement.operands[operand_id] = processed
                break
        else:
            is_negative = operand[0] == "-"
            if is_negative:
                operand = operand[1:]
            if not operand.startswith("r"):
                continue
            register, components = operand.split(".")
            if not (register[0] == "r" and register[1:].isdigit()):
                continue
            var_usages = [None for _ in components]
            for var_name, variable in context.variables.items():
                if variable.register != register:
                    continue
                if (
                    pos[0] in variable.usages
                    and pos[1] in variable.usages[pos[0]]
                    and operand_id in variable.usages[pos[0]][pos[1]]
                ):
                    for comp_idx, component in enumerate(components):
                        if component not in variable.usages[pos[0]][pos[1]][operand_id]:
                            continue
                        var_usages[comp_idx] = (
                            var_name,
                            variable.get_var_comp(component),
                            "-{}" if is_negative else "{}"
                        )
            statement.operands[operand_id] = var_usages


def _substitute_result(statement, pos, context):
    register, components = statement.result.split(".")
    var_usages = [None for _ in components]
    for var_name, variable in context.variables.items():
        if variable.register != register:
            continue
        if pos in variable.modifiers:
            for comp_idx, component in enumerate(components):
                if component not in variable.modifiers[pos]:
                    continue
                var_usages[comp_idx] = (var_name, variable.get_var_comp(component))
    statement.result = var_usages


def _find_usages_in_block(block, register, components, first_statement_id):
    usages = dict()
    for statement_id, statement in enumerate(
        block.statements[first_statement_id:], first_statement_id
    ):
        for operand_id, operand in enumerate(statement.operands):
            if _operand_uses_register(operand, register, components):
                if statement_id not in usages:
                    usages[statement_id] = dict()
                usages[statement_id][operand_id] = copy.copy(components)
        if not isinstance(statement, _AssignStatement):
            continue
        target_reg, target_comp = statement.result.split(".")
        if target_reg != register:
            continue
        components -= set(target_comp)
    return usages, components


def _find_usages(blocks, graph, context, statement_id, block_id, register, components):
    blocks_to_process = [(block_id, statement_id + 1, components)]
    final_usages = dict()
    while len(blocks_to_process) > 0:
        block_id, start_statement, comps = blocks_to_process[0]
        usages, comp_to_process = _find_usages_in_block(
            blocks[block_id], register, comps, start_statement
        )
        if len(usages):
            final_usages[block_id] = usages
        if len(comp_to_process):
            for follow_block_id in graph[block_id].next_blocks:
                blocks_to_process.append((follow_block_id, 0, comp_to_process))
        blocks_to_process = blocks_to_process[1:]
    return final_usages


def _find_parent_block(graph, usages):
    front = usages
    while len(front) > 1:
        last_block = max(front)
        front.remove(last_block)
        front |= set(graph[last_block].prev_blocks)
    return list(front)[0]


def _add_variable(blocks, graph, context, statement_id, block_id):
    register, components = blocks[block_id].statements[statement_id].result.split(".")
    components = set(components)
    current_usages = _find_usages(
        blocks, graph, context, statement_id, block_id, register, copy.copy(components)
    )
    vars_to_merge = set()
    for variable in context.variables.values():
        if variable.register != register or not components & variable.components:
            continue
        mutual_blocks = set(variable.usages.keys()) & set(current_usages.keys())
        for block in mutual_blocks:
            mutual_statements = (
                set(variable.usages[block].keys()) & set(current_usages[block].keys())
            )
            for statement in mutual_statements:
                mutual_operands = (
                    set(variable.usages[block][statement].keys())
                    & set(current_usages[block][statement].keys())
                )
                for operand in mutual_operands:
                    if len(
                        variable.usages[block][statement][operand]
                        & current_usages[block][statement][operand]
                    ):
                        vars_to_merge.add(variable.name)
    if not vars_to_merge:
        init_pos = parent_block = _find_parent_block(graph, set(current_usages.keys()) | {block_id})
        if parent_block == block_id:
            init_pos = (block_id, statement_id)
        context.add_var(current_usages, register, components, init_pos, (block_id, statement_id))
    elif len(vars_to_merge) == 1:
        var_name = list(vars_to_merge)[0]
        context.extend_var(var_name, current_usages, components, graph, (block_id, statement_id))
    else:
        print("Not implemented!")


def _extract_register_name(operand):
    if operand[0] == "-":
        operand = operand[1:]
    return operand.split(".")[0]

def _compute_result_type(statement, context):
    operands_types = []
    for operand in statement.operands:
        if isinstance(operand, list):
            operands_types.extend(context.variables[var_name].type_id for var_name, _, _ in operand)
        else:
            operands_types.append(type_idx.index(operand.type))
    if not all(x != 0 for x in operands_types):
        for operand, type_op in zip(statement.operands, operands_types):
            if type_op == 0:
                print(f"{operand} doesn't have a type!")
    type_evaluators = collections.defaultdict(lambda:max,
        ftou=lambda x: type_idx.index(int),
        utof=lambda x: type_idx.index(float),
        ftoi=lambda x: type_idx.index(int),
        ine=lambda x: type_idx.index(bool),
        ge=lambda x: type_idx.index(bool),
    )
    statement.type_id = type_evaluators[statement.instruction](operands_types)
    context.variables[statement.result[0][0]].type_id = statement.type_id


def _replace_registers_with_variables(
    blocks:list[_CodeBlock],
    graph:list[_GraphNode],
    input_constants,
    header,
    variable_names
):
    variables_context = _VariablesContext(input_constants, header, variable_names)
    for block_id, block in enumerate(blocks):
        for statement_id, statement in enumerate(block.statements):
            if not isinstance(statement, _AssignStatement):
                continue
            _add_variable(blocks, graph, variables_context, statement_id, block_id)



    for block_id, block in enumerate(blocks):
        for statement_id, statement in enumerate(block.statements):
            _substitute_operands(statement, (block_id, statement_id), variables_context)
            if not isinstance(statement, _AssignStatement):
                continue
            _substitute_result(statement, (block_id, statement_id), variables_context)
            _compute_result_type(statement, variables_context)
    for variable in variables_context.variables.values():
        if isinstance(variable.init_pos, tuple):
            blocks[variable.init_pos[0]].statements[variable.init_pos[1]].initializer = True

    for variable in variables_context.variables.values():
        if not isinstance(variable.init_pos, tuple):
            blocks[variable.init_pos].statements = [
                _InitStatement(variable, variable.type_id, len(variable.components))
            ] + blocks[variable.init_pos].statements



def _get_types_for_resources(header):
    resource_types = dict()
    type_casters = {
        "dcl_resource_texture2darray": (lambda x: (x[2], float if "float" in x[1] else int)),
        "dcl_input" : (lambda x: (x[1].split(".")[0], int)),
        "dcl_resource_structured": (lambda x: (x[1], None)),
        "dcl_uav_raw": (lambda x: (x[1], int)),
        "dcl_uav_structured": (lambda x: (x[1], float)),
    }
    for line in header:
        tokens = line.split()
        if tokens[0] in type_casters:
            res_name, res_type = type_casters[tokens[0]](tokens)
            resource_types[res_name] = res_type
        else:
            print(line, "isn't processed")
    return resource_types


def _remove_extra_components_for_statement(statement):
    result_components = statement.result.split(".")[1]
    def single_operand_reductor(operand):
        if operand[0] == "l" or "." not in operand:
            return operand
        reg, comps = operand.split(".")
        if len(result_components) == len(comps):
            return operand
        comps = "".join(map(lambda x: comps["xyzw".index(x)], result_components))
        return f"{reg}.{comps}"

    def components_count_setter(operand, count):
        reg, comps = operand.split(".")
        return f"{reg}.{comps[:count]}"

    def default_components_reducer(operands):
        return [single_operand_reductor(operand) for operand in operands]
    reducers = collections.defaultdict(lambda : default_components_reducer)
    reducers["ld_indexable(texture2darray)(float,float,float,float)"] = lambda x: [
        components_count_setter(x[0], 3), single_operand_reductor(x[1])
    ]
    reducers["dp2"] = lambda x: [components_count_setter(x[0], 2), components_count_setter(x[1], 2)]
    reducers["dp3"] = lambda x: [components_count_setter(x[0], 3), components_count_setter(x[1], 3)]
    reducers["dp4"] = lambda x: [components_count_setter(x[0], 4), components_count_setter(x[1], 4)]
    statement.operands = reducers[statement.instruction](statement.operands)

def _remove_extra_components(statements):
    for statement in statements:
        if isinstance(statement, _AssignStatement):
            _remove_extra_components_for_statement(statement)

def _gen_hlsl(blocks):
    hlsl = ""
    for block in blocks:
        for statement in block.statements:
            depth = block.depth
            if isinstance(statement, _FlowControlStatement):
                depth += statement.get_depth()
            hlsl += "  " * depth +  str(statement) + "\n"
    return hlsl

def recover(disasm: str, inputs, variable_names) -> str:
    """Main function, call it to recover HLSL"""
    # Filter statements
    header, raw_code = _filter_disasm_statements(disasm)
    # Split statement on tokens
    raw_tokens = [_statement_str_to_tokens(line) for line in raw_code]
    # Convert list of tokens to statement object
    statements = _build_statements(raw_tokens)
    _remove_extra_components(statements)
    # Split code on blocks
    blocks, previous_block_links = _split_commands_on_blocks(statements)
    # Create a graph of blocks
    graph = _gen_graph(previous_block_links)
    # Replace registers with variables
    _replace_registers_with_variables(blocks, graph, inputs, header, variable_names)
    return _gen_hlsl(blocks)
    # Substitute common functions
    # Substitute one place variables
    # Return generated code
