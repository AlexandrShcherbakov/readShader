import collections
import copy

type_names = ["UNKNOWN", "bool", "int", "int", "uint", "float"]
_FLOAT = 5
_UINT = 4
_HEX = 3
_INT = 2
_BOOL = 1
_UNKNOWN = 0

def _squash_brackets_expression(tokens):
    def squash_brackets(tokens, begin, end, sep):
        i = len(tokens) - 1
        while i >= 0:
            while begin in tokens[i] and end not in tokens[i]:
                tokens[i] += sep + tokens[i + 1]
                del tokens[i + 1]
            i -= 1
    
    squash_brackets(tokens, "(", ")", ",")
    squash_brackets(tokens, "[", "]", "")

replace_table_operations = {
    "utof": ("asfloat({})", True),
    "itof": ("asfloat({})", True),
    "mad": ("{} * {} + {}", False),
    "ftou": ("asuint({})", True),
    "mov": ("{}", True),
    "ld_indexable(texture2darray)(float,float,float,float)": ("{1.name}[{0}].{1.components}", True),
    "ftoi": ("asint({})", True),
    "ine": ("{} != {}", False),
    "ne": ("{} != {}", False),
    "eq": ("{} == {}", False),
    "ieq": ("{} == {}", False),
    "lt": ("{} < {}", False),
    "mul": ("{} * {}", True),
    "ge": ("{} >= {}", False),
    "movc": ("{} ? {} : {}", False),
    "div": ("{} / {}", True),
    "frc": ("frac({})", True),
    "ld_structured_indexable(structured_buffer,stride=16)(mixed,mixed,mixed,mixed)": ("{2.name}[{0} + {1}].{2.components}", True),
    "and": ("{} & {}", False),
    "add": ("{} + {}", False),
    "dp3": ("dot({}, {})", True),
    "dp2": ("dot({}, {})", True),
    "dp4": ("dot({}, {})", True),
    "sqrt": ("sqrt({})", True),
    "rsq": ("rsqrt({})", True),
    "round_ni": ("floor({})", True),
    "round_z": ("trunc({})", True),
    "exp": ("exp({})", True),
    "min": ("min({}, {})", True),
    "max": ("max({}, {})", True),
    "mul_sat": ("saturate({} * {})", True),
    "div_sat": ("saturate({} / {})", True),
    "ld_indexable": ("{1.name}[{0}].{1.components}", True),
    "ld_structured": ("{2.name}[{0} + {1}].{2.components}", True),
    "sample_l": ("{1.name}.SampleLevel({2}, {0}, {3}).{1.components}", True),
    "sample_indexable(texture2d)(float,float,float,float)": ("{1.name}.Sample({2}, {0}).{1.components}", True),
    "mad_sat": ("saturate({} * {} + {})", True),
    "add_sat": ("saturate({} + {})", True),
    "log": ("log({})", True),
    "sample_l(texturecube)(float,float,float,float)": ("{1.name}.SampleLevel({2}, {0}, {3}).{1.components}", True),
    "mov_sat": ("saturate({})", True),
    "initializer": ("", True),
}

arg_sizes = {
    "ld_indexable(texture2darray)(float,float,float,float)": 3,
    "ld_structured": 1,
    "sample_l(texturecube)(float,float,float,float)": 3,
}

replace_table_special = {
    "atomic_iadd": "InterlockedAdd({0.name}[{1}], {2})",
    "store_structured": "{0.name}[{1} + {2}].{0.components} = {3}",
    "resinfo_indexable(texturecube)(float,float,float,float)_float": "{2.name}.GetDimensions({1}, {0})",
    "sincos": "sincos({}, {}, {})"
}

replace_table_operators = {
    "if_nz": "if ({}) {{",
    "else": "}} else {{",
    "endif": "}}",
    "ret": "return",
    "discard_nz": "if ({}) discard;",
}

scope_modifiers = {
    "if_nz": lambda x: x.enter_scope(),
    "else": lambda x: x.switch_scope(),
    "endif": lambda x: x.exit_scope(),
}

def _default_type_extr(operands):
    tp = _UNKNOWN
    for t in operands:
        if isinstance(t, (_Variable, _VariableAccessor, _ShaderInput, _Constant)):
            tp = max(tp, t.type)
    return tp

type_by_instr = collections.defaultdict(lambda: _default_type_extr)
type_by_instr.update({
    "utof": lambda args: _FLOAT,
    "ftou": lambda args: _UINT,
    "ftoi": lambda args: _INT,
    "ld_indexable(texture2darray)(float,float,float,float)": lambda args: _FLOAT,
    "ld_indexable": lambda args: args[1].type,
    "ge": lambda args: _BOOL,
    "ine": lambda args: _BOOL,
    "ne": lambda args: _BOOL,
    "eq": lambda args: _BOOL,
    "ieq": lambda args: _BOOL,
    "lt": lambda args: _BOOL,
})

special_res_components = {
    "dp2": 2,
    "dp3": 3,
    "dp4": 4,
}

class _Variable:
    def __init__(self, name, register, components, tp, const_value=None):
        self.name = name
        self.register = register
        self.components = components
        self.type = tp
        self.const_value = const_value

    def __str__(self):
        return self.name

    def get_type(self):
        if len(self.components) == 1:
            return type_names[self.type]
        return f"{type_names[self.type]}{len(self.components)}"


class ShaderOutput:
    def __init__(self, name, components, tp = None):
        self.name = name
        self.components = components
        # self.type = tp

    def __str__(self):
        return self.name

    # def get_type(self):
    #     if len(self.components) == 1:
    #         return type_names[self.type]
    #     return f"{type_names[self.type]}{len(self.components)}"

class _VariableAccessor:
    def __init__(self, variable, components, wrappers):
        self.variable = variable
        self.components = components
        self.type = self.variable.type
        self.wrappers = wrappers
    
    def tune_components(self, res, command):
        if len(res.components) == 1:
            if len(self.components) == 1:
                return
        if command in arg_sizes:
            self.components = self.components[:arg_sizes[command]]
        else:
            self.components = "".join([self.components["xyzw".index(c)] for c in res.components])

    def whole_value(self):
        return "xyzw".startswith(self.components) and len(self.components) == len(self.variable.components)

    def __str__(self):
        comps = f".{self.components}"
        if self.whole_value():
            comps = ""
        return f"{self.wrappers[0]}{self.variable}{comps}{self.wrappers[1]}"

    def __eq__(self, a):
        return type(self) == type(a) and self.variable == a.variable and self.components == a.components and self.wrappers == a.wrappers


class _CombinedValue:
    def __init__(self, values, wrappers):
        self.values = values
        self.type = max([val[2].type for val in values])
        self.wrappers = wrappers

    def tune_components(self, res, command):
        if command in arg_sizes:
            self.values = self.values[:arg_sizes[command]]
        else:
            self.values = [self.values["xyzw".index(c)] for c in res.components]

    def _get_comp_access(self, value):
        if len(value) == 1:
            return value[2].name
        if value[2].const_value:
            return str(value[2].const_value.values["xyzw".index(value[1])])
        return f"{value[2].name}.{value[1]}"

    def __str__(self):
        vals = [
            self._get_comp_access(val) for val in self.values
        ]
        return f"{self.wrappers[0]}{type_names[self.type]}{len(self.values)}({', '.join(vals)}){self.wrappers[1]}"


class _Context:
    def __init__(self, external_inputs, internal_inputs):
        self.variables_map = [dict()]
        self.stack = [0]
        self.variables_count = 0
        self.external_inputs = external_inputs
        self.internal_inputs = internal_inputs
        self.tabs = 0
        self.replacer = dict()
        self.variables = dict()

    def add_variable(self, reg, tp, const_value, variable_name = None):
        if variable_name in self.variables:
            self.variables[variable_name].type = tp
            return self.variables[variable_name]
        reg_name, components = reg.split('.')
        if variable_name is None:
            variable_name = f"var_{self.variables_count}"
            self.variables_count += 1
        var = _Variable(variable_name, reg_name, components, tp, const_value)
        self.variables[variable_name] = var
        self.variables_map[self.stack[-1]].update({f"{reg_name}.{components[i]}": (variable_name, "xyzw"[i], var) for i in range(len(components))})
        return var

    def get_variable_by_reg(self, reg, components, wrappers):
        in_other_scopes = []
        variables = []
        for comp in components:
            key = f"{reg}.{comp}"
            for layer in reversed(self.stack):
                if key not in self.variables_map[layer]:
                    continue
                variables.append(self.variables_map[layer][key])
                break
            else:
                var = None
                for layer in range(self.stack[-1], len(self.variables_map)):
                    if key in self.variables_map[layer]:
                        if var:
                            self.replacer[var.name].append(self.variables_map[layer][key][2])
                            continue
                        var = self.variables_map[layer][key][2]
                        variables.append(self.variables_map[layer][key])
                        self.replacer[var.name] = [var]



        if len({var_name for var_name, var_comp, var in variables}) == 1:
            return _VariableAccessor(variables[0][2], "".join([var_comp for var_name, var_comp, var in variables]), wrappers)
        return _CombinedValue(variables, wrappers)

    def enter_scope(self):
        self.stack.append(len(self.variables_map))
        self.variables_map.append(dict())
        self.tabs += 1

    def switch_scope(self):
        self.stack[-1] = len(self.variables_map)
        self.variables_map.append(dict())

    def exit_scope(self):
        self.stack = self.stack[:-1]
        self.tabs -= 1


def _tune_operands(res, command, operands):
    if command in special_res_components:
        res = copy.deepcopy(res)
        res.components = "xyzw"[:special_res_components[command]]
    for i in range(len(operands)):
        operands[i].tune_components(res, command)


class _ResStatement:
    def __init__(self, command, res, operands, context, initialize=True, variable_name=None):
        self.init_command = command
        self.command, self.inplace_without_brackets = replace_table_operations[command]
        # Parse operands
        self.operands = _parse_operands(operands, context)
        const_value = None
        if command == "mov" and len(self.operands) == 1 and isinstance(self.operands[0], _Constant):
            const_value = self.operands[0]
        # Evaluate result variable
        if res[0] == "o":
            self.res = ShaderOutput(*res.split("."))
            initialize = False
        else:
            self.res = context.add_variable(res, type_by_instr[command](self.operands), const_value, variable_name)
        # Tune operands
        _tune_operands(self.res, command, self.operands)

        self.tabs = context.tabs
        self.initialize = initialize
        self.scope_id = context.stack[-1]

    def __str__(self):
        prefix = self.res.get_type() + " " if self.initialize else ""
        if len(self.command) == 0:
            return f"{'  ' * self.tabs}{prefix}{self.res};"
        return f"{'  ' * self.tabs}{prefix}{self.res} = {self.command.format(*self.operands)};"


class _TmpResStatement:
    def __init__(self, init_statement):
        self.statement = init_statement
    
    def __str__(self):
        value = f"{self.statement.command.format(*self.statement.operands)}"
        if self.statement.inplace_without_brackets:
            return value
        return f"({value})"

class _Statement:
    def __init__(self, command, operands, context):
        self.command = replace_table_operators[command]
        self.operands = _parse_operands(operands, context)
        self.tabs = context.tabs
        if command in scope_modifiers:
            scope_modifiers[command](context)
        if command == "else" or command == "endif":
            self.tabs -= 1
        self.scope_id = context.stack[-1]
        if command.startswith("if"):
            self.scope_id = context.stack[-2]

    def __str__(self):
        return '  ' * self.tabs + self.command.format(*self.operands)


class _StoreStatement:
    def __init__(self, command, operands, context):
        self.command = replace_table_special[command]
        self.operands = _parse_operands(operands, context)
        self.tabs = context.tabs
        _tune_operands(self.operands[0], command, [self.operands[-1]])
        self.scope_id = context.stack[-1]

    def __str__(self):
        return '  ' * self.tabs + self.command.format(*self.operands) + ";"


class _Initializer:
    def __init__(self, variable, context):
        self.variable = variable

    def __str__(self):
        return f"{self.variable.get_type()} {self.variable.name};"

class _ShaderInput:
    def __init__(self, name, tp, wrappers):
        self.name = name
        self.type = tp
        self.components = "xyzw"
        self.wrappers = wrappers

    def tune_components(self, res, command):
        self.components = res.components

    def __str__(self):
        return f"{self.wrappers[0]}{self.name}.{self.components}{self.wrappers[1]}"


class _Constant:
    def __init__(self, values):
        vals = values.split(',')
        self.type = _FLOAT if '.' in values else (_INT if '-' in values else _UINT)
        if self.type == _FLOAT:
            caster = float
        elif 'x' in vals[0]:
            caster = lambda x: int(x, 16)
            self.type = _HEX
        else:
            caster = int
        self.values = list(map(caster, vals))

    def tune_components(self, res, command):
        if len(self.values) == 1:
            return
        self.values = [self.values["xyzw".index(c)] for c in res.components]

    def __str__(self):
        if len(self.values) == 1:
            if isinstance(self.values[0], int):
                return hex(self.values[0])
            return str(self.values[0])
        return f"{type_names[self.type]}{len(self.values)}({', '.join(list(map(str, self.values)))})"


class _CbAccessor:
    def __init__(self, cb_idx, variable, offset, components):
        self.cb_idx = cb_idx
        self.variable_accessor = variable
        self.components = components
        self.offset = offset

    def tune_components(self, res, command):
        self.components = "".join([self.components["xyzw".index(c)] for c in res.components])

    def __str__(self):
        return f"cb{self.cb_idx}[{self.variable_accessor} + {self.offset}].{self.components}"


class _Tex2DArray:
    def __init__(self, name, tp):
        self.name = name
        self.type = tp

    def get_usage_class(self):
        return _Tex2DArrayUsage

class _Tex2D:
    def __init__(self, name, tp):
        self.name = name
        self.type = tp

    def get_usage_class(self):
        return _Tex2DUsage

class _TexCube:
    def __init__(self, name, tp):
        self.name = name
        self.type = tp

    def get_usage_class(self):
        return _TexCubeUsage

class _Tex3D:
    def __init__(self, name, tp):
        self.name = name
        self.type = tp

    def get_usage_class(self):
        return _Tex3DUsage


class _Sampler:
    def __init__(self, name):
        self.name = name

    def get_usage_class(self):
        return _SamplerUsage

    def get_components(self):
        return None

class _SBuffer:
    def __init__(self, name, tp):
        self.name = name
        self.type = tp

    def get_usage_class(self):
        return _SBufferUsage

class _UAVSBuffer:
    def __init__(self, name, tp):
        self.name = name
        self.type = tp

    def get_usage_class(self):
        return _UAVSBufferUsage

class _UAV:
    def __init__(self, name, tp):
        self.name = name
        self.type = tp

    def get_usage_class(self):
        return _UAVUsage

    def get_components(self):
        return "x"

class _Tex2DArrayUsage:
    def __init__(self, array, components):
        self.name = array.name
        self.type = array.type
        self.components = components

    def tune_components(self, res, command):
        self.components = "".join([self.components["xyzw".index(c)] for c in res.components])

class _Tex2DUsage:
    def __init__(self, array, components):
        self.name = array.name
        self.type = array.type
        self.components = components

    def tune_components(self, res, command):
        self.components = "".join([self.components["xyzw".index(c)] for c in res.components])

class _TexCubeUsage:
    def __init__(self, array, components):
        self.name = array.name
        self.type = array.type
        self.components = components

    def tune_components(self, res, command):
        self.components = "".join([self.components["xyzw".index(c)] for c in res.components])

class _Tex3DUsage:
    def __init__(self, array, components):
        self.name = array.name
        self.type = array.type
        self.components = components

    def tune_components(self, res, command):
        self.components = "".join([self.components["xyzw".index(c)] for c in res.components])

class _SamplerUsage:
    def __init__(self, array, components):
        self.name = array.name

    def tune_components(self, res, command):
        pass

    def __str__(self):
        return self.name

class _SBufferUsage:
    def __init__(self, array, components):
        self.name = array.name
        self.type = array.type
        self.components = components

    def tune_components(self, res, command):
        self.components = "".join([self.components["xyzw".index(c)] for c in res.components])

class _UAVSBufferUsage:
    def __init__(self, array, components):
        self.name = array.name
        self.type = array.type
        self.components = components

    def tune_components(self, res, command):
        self.components = "".join([self.components["xyzw".index(c)] for c in res.components])

class _UAVUsage:
    def __init__(self, array, components):
        self.name = array.name
        self.type = array.type
        self.components = components

    def tune_components(self, res, command):
        self.components = "".join([self.components["xyzw".index(c)] for c in res.components])


SHADER_INPUT_NAMES = {
    "vThreadID" : _UINT,
}


def _parse_operands(operands, context):
    parsed = []
    for operand in operands:
        wrappers = ("", "")
        if operand.startswith('-'):
            wrappers = ("-", "")
            operand = operand[1:]
        if operand.startswith("abs("):
            wrappers = ("abs(", ")")
            operand = operand[4:-1]

        if operand[0] == 'r' and operand[1:].split('.')[0].isdigit():
            parsed.append(context.get_variable_by_reg(*operand.split('.'), wrappers))
        elif operand.rsplit('.', 1)[0] in SHADER_INPUT_NAMES:
            parsed.append(_ShaderInput(operand.rsplit('.', 1)[0], SHADER_INPUT_NAMES[operand.rsplit('.', 1)[0]], wrappers))
        elif operand.rsplit('.', 1)[0] in context.external_inputs:
            parsed.append(_ShaderInput(operand.rsplit('.', 1)[0], context.external_inputs[operand.rsplit('.', 1)[0]], wrappers))
        elif operand.split('.')[0] in context.internal_inputs:
            inp = context.internal_inputs[operand.split('.')[0]]
            components = operand.split('.')[1] if "." in operand else inp.get_components()
            parsed.append(inp.get_usage_class()(inp, components))
        elif operand.startswith("l("):
            parsed.append(_Constant(operand[2: -1]))
        elif operand.startswith("cb") and "r" in operand:
            cb_idx = int(operand[2:operand.index("[")])
            variable = context.get_variable_by_reg(*operand[operand.index("[") + 1: operand.index("+")].split('.'), ("", ""))
            offset = int(operand[operand.index("+") + 1: operand.index("]")])
            components = operand.split(".")[-1]
            parsed.append(_CbAccessor(cb_idx, variable, offset, components))
        else:
            raise Exception(f"Unknown operand {operand}")
    return parsed


def dxil_transform(dxil, inputs, var_names):
    dxil_lines = []
    input_lines = []
    for line in dxil.split('\n'):
        if line.strip(' ').startswith("dcl_"):
            input_lines.append(line)
        if not line.split(":")[0].strip().isdigit():
            continue
        command = line.split(":")[1]
        dxil_lines.append(list(filter(lambda x: len(x) > 0, map(lambda x: x.strip(', '), command.split(' ')))))
    blocks = split_commands_on_blocks(dxil_lines)
    add_reversed_links(blocks)
    mark_initializers_in_blocks(blocks)
    code, initializers, variable_names = regenerate_code_and_initializers(blocks)
    return dxil_to_commands(input_lines, code, initializers, variable_names, inputs, var_names)

def process_inputs(inputs):
    res = dict()
    for inp in inputs:
        inp = inp.strip(' ').split()
        if inp[0] == "dcl_resource_texture2darray":
            res[inp[2]] = _Tex2DArray(inp[2], type_names.index(inp[1][1:].split(',')[0]))
        elif inp[0] == "dcl_resource_texture2d":
            res[inp[2]] = _Tex2D(inp[2], type_names.index(inp[1][1:].split(',')[0]))
        elif inp[0] == "dcl_resource_texturecube":
            res[inp[2]] = _TexCube(inp[2], type_names.index(inp[1][1:].split(',')[0]))
        elif inp[0] == "dcl_resource_structured":
            res[inp[1]] = _SBuffer(inp[1], _UNKNOWN)
        elif inp[0] == "dcl_uav_raw":
            res[inp[1]] = _UAV(inp[1], _UNKNOWN)
        elif inp[0] == "dcl_uav_structured":
            res[inp[1]] = _UAVSBuffer(inp[1], _UNKNOWN)
        elif inp[0] == "dcl_resource_texture3d":
            res[inp[2]] = _Tex3D(inp[2], type_names.index(inp[1][1:].split(',')[0]))
        elif inp[0] == "dcl_sampler":
            res[inp[1]] = _Sampler(inp[1])
    return res


def replace_known_patterns(lines, var_usages):
    to_erase = []
    for i in range(len(lines) - 1):
        if (
            isinstance(lines[i], _ResStatement) and lines[i].init_command.startswith("dp") and lines[i].operands[0] == lines[i].operands[1]
            and var_usages[lines[i].res.name] == 1
            and isinstance(lines[i + 1], _ResStatement) and lines[i + 1].init_command == "sqrt" and isinstance(lines[i + 1].operands[0], _VariableAccessor)
            and lines[i + 1].operands[0].whole_value()
            and lines[i].res == lines[i + 1].operands[0].variable
        ):
            lines[i].command = "length({})"
            lines[i].operands = lines[i].operands[:1]
            lines[i].res = lines[i + 1].res
            to_erase.append(i + 1)
    
    for i in sorted(to_erase, reverse=True):
        lines = lines[:i] + lines[i + 1:]
    return lines


def reduce_lines_count(lines, var_usages):
    to_erase = []
    for i in range(len(lines) - 1):
        if not isinstance(lines[i], (_ResStatement, _StoreStatement, _Statement)):
            continue
        for k in range(len(lines[i].operands)):
            if isinstance(lines[i].operands[k], _VariableAccessor) and var_usages[lines[i].operands[k].variable.name] == 1:
                for j in range(i):
                    if isinstance(lines[j], _ResStatement) and lines[j].res == lines[i].operands[k].variable and lines[j].scope_id == lines[i].scope_id:
                        lines[i].operands[k] = _TmpResStatement(lines[j])
                        to_erase.append(j)
                        break
    
    for i in sorted(to_erase, reverse=True):
        lines = lines[:i] + lines[i + 1:]
    return lines


def dxil_to_commands(input_lines, dxil_lines, initializers, variable_names, external_inputs, var_names):
    processed = []
    context = _Context(external_inputs, process_inputs(input_lines))
    for command, is_init, variable_name in zip(dxil_lines, initializers, variable_names):
        _squash_brackets_expression(command)
        # Classify command
        if command[0] in replace_table_operations:
            processed.append(_ResStatement(command[0], command[1], command[2:], context, is_init, variable_name))
        elif command[0] in replace_table_operators:
            processed.append(_Statement(command[0], command[1:], context))
        elif command[0] in replace_table_special:
            processed.append(_StoreStatement(command[0], command[1:], context))
        else:
            raise Exception(f"Unsupported command {command}")
        # print(processed[-1])

    for key, variables in context.replacer.items():
        names = {v.name for v in variables}
        processed = [_Initializer(variables[0], context)] + processed
        for p in processed:
            if isinstance(p, _ResStatement) and p.res.name in names:
                p.initialize = False
                p.res = variables[0]

    for i in reversed(range(len(processed))):
        if isinstance(processed[i], _ResStatement) and isinstance(processed[i].res, _Variable) and processed[i].res.const_value:
            del processed[i]
    
    variable_usages = collections.defaultdict(lambda:0)
    for st in processed:
        if not isinstance(st, (_ResStatement, _Statement, _StoreStatement)):
            continue
        for operand in st.operands:
            if isinstance(operand, _VariableAccessor):
                variable_usages[operand.variable.name] += 1
            elif isinstance(operand, _CombinedValue):
                for v in operand.values:
                    variable_usages[v[0]] += 1
        if isinstance(st, _ResStatement) and not st.initialize:
            variable_usages[st.res.name] += 1

    processed = replace_known_patterns(processed, variable_usages)
    processed = reduce_lines_count(processed, variable_usages)

    for i in range(len(processed)):
        if not isinstance(processed[i], _ResStatement):
            continue
        if processed[i].res.name in var_names:
            processed[i].res.name = var_names[processed[i].res.name]

    return "\n".join(list(map(str, processed)))


class CodeBlock:
    def __init__(self, layer, previous_blocks):
        self.instructions = []
        self.layer = layer
        self.previous_blocks = previous_blocks
        self.initializer = []
        self.var_names = []

    def add_instruction(self, instruction, init=None):
        self.instructions.append(instruction)
        self.initializer.append(init)
        self.var_names.append("UNNAMED")

    def __repr__(self):
        return str(self.instructions)


def split_commands_on_blocks(commands):
    current_layer = 0
    blocks = [CodeBlock(current_layer, [])]
    FLOW_CONTROLLERS = { "if_nz": 1, "else": 0, "endif": -1 }
    for command in commands:
        blocks[-1].add_instruction(command)
        if command[0] not in FLOW_CONTROLLERS:
            continue
        current_layer += FLOW_CONTROLLERS[command[0]]
        if blocks[-1].layer == current_layer - 1:
            previous_blocks = [len(blocks) - 1]
        elif blocks[-1].layer == current_layer:
            for idx in range(len(blocks) - 2, -1, -1):
                if blocks[idx].layer == current_layer - 1:
                    previous_blocks = [idx]
                    break
        elif blocks[-1].layer == current_layer + 1:
            previous_blocks = []
            for idx in range(len(blocks) - 2, -1, -1):
                if blocks[idx].layer == blocks[-1].layer and blocks[idx + 1].layer == blocks[idx].layer:
                    previous_blocks.append(idx)
                    break
            previous_blocks.append(len(blocks) - 1)
        blocks.append(CodeBlock(current_layer, previous_blocks))
    return blocks


def add_reversed_links(blocks):
    for block_id, block in enumerate(blocks):
        block.next_blocks = []
        for prev_block_id in block.previous_blocks:
            blocks[prev_block_id].next_blocks.append(block_id)


def create_context_for_block(blocks, block_start_id, register_name, register_components, var_name):
    context = {block_start_id}
    layer = {block_start_id}
    next_layer = set()
    register_components = set(register_components)
    while len(layer):
        context |= layer
        for block_id in layer:
            next_layer |= set(blocks[block_id].next_blocks)
        if len(layer) == 1 and blocks[list(layer)[0]].layer == 0:
            for statement_id, statement in enumerate(blocks[list(layer)[0]].instructions):
                if statement[0] not in replace_table_operations:
                    continue
                target_register, target_components = statement[1].split(".", 1)
                if target_register != register_name:
                    continue
                register_components = register_components - set(target_components)
                if len(register_components):
                    blocks[list(layer)[0]].initializer[statement_id] = False
                    blocks[list(layer)[0]].var_names[statement_id] = var_name
        if not len(register_components):
            break
        layer = next_layer
        next_layer = set()
    return context


def find_parent_block(blocks, context):
    for block_id in context:
        if all([next_block_id not in context for next_block_id in blocks[block_id].next_blocks]):
            break
    current_layer = set(blocks[block_id].previous_blocks)
    depth = min(blocks[block_id].layer for block_id in context)
    while len(current_layer) != 1 and blocks[list(current_layer)[0]].layer != depth:
        next_layer = set()
        for idx in current_layer:
            next_layer |= set(blocks[idx].previous_blocks)
        current_layer = next_layer
        next_layer = set()
    return list(current_layer)[0]


def mark_initializers_in_blocks(blocks):
    variables_count = 0
    for block_idx, block in enumerate(blocks):
        for statement_idx, statement in enumerate(block.instructions):
            if statement[0] not in replace_table_operations:
                continue
            if block.initializer[statement_idx] is not None:
                continue
            var_name = "var_" + str(variables_count)
            variables_count += 1
            block.var_names[statement_idx] = var_name
            if block.layer == 0:
                block.initializer[statement_idx] = True
                continue
            register, components = statement[1].split(".", 1)
            context = create_context_for_block(blocks, block_idx, register, components, var_name)
            parent_block = find_parent_block(blocks, context)
            if parent_block == block_idx:
                block.initializer[statement_idx] = True
            else:
                blocks[parent_block].instructions = [["initializer", statement[1]]] + blocks[parent_block].instructions
                blocks[parent_block].initializer = [True] + blocks[parent_block].initializer
                blocks[parent_block].var_names = [var_name] + blocks[parent_block].var_names
                block.initializer[statement_idx] = False


def regenerate_code_and_initializers(blocks):
    code = []
    initializers = []
    var_names = []
    for block in blocks:
        code += block.instructions
        initializers += block.initializer
        var_names += block.var_names
    return code, initializers, var_names
