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
    i = len(tokens) - 1
    while i >= 0:
        while "(" in tokens[i] and ")" not in tokens[i]:
            tokens[i] += ',' + tokens[i + 1]
            del tokens[i + 1]
        i -= 1

replace_table_operations = {
    "utof": "asfloat({})",
    "mad": "{} * {} + {}",
    "ftou": "asuint({})",
    "mov": "{}",
    "ld_indexable(texture2darray)(float,float,float,float)": "{1.name}[{0}].{1.components}",
    "ftoi": "asint({})",
    "ine": "{} != {}",
    "mul": "{} * {}",
    "ge": "{} >= {}",
    "movc": "{} ? {} : {}",
    "div": "{} / {}",
    "frc": "frac({})",
    "ld_structured_indexable(structured_buffer,stride=16)(mixed,mixed,mixed,mixed)": "{2.name}[{0} + {1}].{2.components}",
    "and": "{} & {}",
    "add": "{} + {}",
    "dp3": "dot({}, {})",
    "sqrt": "sqrt({})",
    "min": "min({}, {})",
}

arg_sizes = {
    "ld_indexable(texture2darray)(float,float,float,float)": 3
}

replace_table_special = {
    "atomic_iadd": "InterlockedAdd({0.name}[{1}], {2})",
    "store_structured": "{0.name}[{1} + {2}].{0.components} = {3}"
}

replace_table_operators = {
    "if_nz": "if ({}) {{",
    "else": "}} else {{",
    "endif": "}}",
    "ret": "return",
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
    "ge": lambda args: _BOOL,
    "ine": lambda args: _BOOL,
})

special_res_components = {
    "dp2": 2,
    "dp3": 3,
    "dp4": 4,
}

class _Variable:
    def __init__(self, name, register, components, tp):
        self.name = name
        self.register = register
        self.components = components
        self.type = tp

    def __str__(self):
        return self.name

    def get_type(self):
        if len(self.components) == 1:
            return type_names[self.type]
        return f"{type_names[self.type]}{len(self.components)}"


class _VariableAccessor:
    def __init__(self, variable, components, negative):
        self.variable = variable
        self.components = components
        self.type = self.variable.type
        self.negative = negative
    
    def tune_components(self, res, command):
        if len(res.components) == 1:
            if len(self.components) == 1:
                return
        self.components = "".join([self.components["xyzw".index(c)] for c in res.components])

    def __str__(self):
        comps = f".{self.components}"
        if "xyzw".startswith(self.components) and len(self.components) == len(self.variable.components):
            comps = ""
        return f"{'-' if self.negative else ''}{self.variable}{comps}"


class _CombinedValue:
    def __init__(self, values, negative):
        self.values = values
        self.type = max([val[2].type for val in values])
        self.negative = negative

    def tune_components(self, res, command):
        if command in arg_sizes:
            self.values = self.values[:arg_sizes[command]]
        else:
            self.values = [self.values["xyzw".index(c)] for c in res.components]

    def __str__(self):
        vals = [
            (f'{var_name}.{var_comp}' if len(var.components) > 1 else var_name) for var_name, var_comp, var in self.values
        ]
        return f"{'-' if self.negative else ''}{type_names[self.type]}{len(self.values)}({', '.join(vals)})"


class _Context:
    def __init__(self, external_inputs, internal_inputs):
        self.variables_map = [dict()]
        self.stack = [0]
        self.variables_count = 0
        self.external_inputs = external_inputs
        self.internal_inputs = internal_inputs
        self.tabs = 0
        self.replacer = dict()

    def add_variable(self, reg, tp):
        reg_name, components = reg.split('.')
        variable_name = f"var_{self.variables_count}"
        self.variables_count += 1
        var = _Variable(variable_name, reg_name, components, tp)
        self.variables_map[self.stack[-1]].update({f"{reg_name}.{components[i]}": (variable_name, "xyzw"[i], var) for i in range(len(components))})
        return var
    
    def get_variable_by_reg(self, reg, components, negative=False):
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
            return _VariableAccessor(variables[0][2], "".join([var_comp for var_name, var_comp, var in variables]), negative)
        return _CombinedValue(variables, negative)

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
    def __init__(self, command, res, operands, context, initialize=True):
        self.command = replace_table_operations[command]
        # Parse operands
        self.operands = _parse_operands(operands, context)
        # Evaluate result variable
        self.res = context.add_variable(res, type_by_instr[command](self.operands))
        # Tune operands
        _tune_operands(self.res, command, self.operands)

        self.tabs = context.tabs
        self.initialize = initialize

    def __str__(self):
        prefix = self.res.get_type() + " " if self.initialize else ""
        return f"{'  ' * self.tabs}{prefix}{self.res} = {self.command.format(*self.operands)}"


class _Statement:
    def __init__(self, command, operands, context):
        self.command = replace_table_operators[command]
        self.operands = _parse_operands(operands, context)
        self.tabs = context.tabs
        if command in scope_modifiers:
            scope_modifiers[command](context)
        if command == "else" or command == "endif":
            self.tabs -= 1

    def __str__(self):
        return '  ' * self.tabs + self.command.format(*self.operands)


class _StoreStatement:
    def __init__(self, command, operands, context):
        self.command = replace_table_special[command]
        self.operands = _parse_operands(operands, context)
        self.tabs = context.tabs
        _tune_operands(self.operands[0], command, [self.operands[-1]])

    def __str__(self):
        return '  ' * self.tabs + self.command.format(*self.operands)


class _Initializer:
    def __init__(self, variable, context):
        self.variable = variable

    def __str__(self):
        return f"{self.variable.get_type()} {self.variable.name}"

class _ShaderInput:
    def __init__(self, name, tp, negative):
        self.name = name
        self.type = tp
        self.components = "xyzw"
        self.negative = negative

    def tune_components(self, res, command):
        self.components = res.components

    def __str__(self):
        return f"{'-' if self.negative else ''}{self.name}.{self.components}"


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
        if len(res.components) == 1:
            if len(self.values) == 1:
                return
        self.values = [self.values["xyzw".index(c)] for c in res.components]

    def __str__(self):
        if len(self.values) == 1:
            return str(self.values[0])
        return f"{type_names[self.type]}{len(self.values)}({', '.join(list(map(str, self.values)))})"


class _Tex2DArray:
    def __init__(self, name, tp):
        self.name = name
        self.type = tp

    def get_usage_class(self):
        return _Tex2DArrayUsage

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
        negative = operand.startswith('-')
        if negative:
            operand = operand[1:]

        if operand[0] == 'r' and operand[1:].split('.')[0].isdigit():
            parsed.append(context.get_variable_by_reg(*operand.split('.'), negative))
        elif operand.split('.')[0] in SHADER_INPUT_NAMES:
            parsed.append(_ShaderInput(operand.split('.')[0], SHADER_INPUT_NAMES[operand.split('.')[0]], negative))
        elif operand.split('.')[0] in context.external_inputs:
            parsed.append(_ShaderInput(operand.split('.')[0], context.external_inputs[operand.split('.')[0]], negative))
        elif operand.split('.')[0] in context.internal_inputs:
            inp = context.internal_inputs[operand.split('.')[0]]
            components = operand.split('.')[1] if "." in operand else inp.get_components()
            parsed.append(inp.get_usage_class()(inp, components))
        elif operand.startswith("l("):
            parsed.append(_Constant(operand[2: -1]))
        else:
            raise Exception(f"Unknown operand {operand}")
    return parsed


def dxil_transform(dxil, inputs):
    dxil_lines = []
    input_lines = []
    for line in dxil.split('\n'):
        if line.strip(' ').startswith("dcl_"):
            input_lines.append(line)
        if not line.split(":")[0].strip().isdigit():
            continue
        command = line.split(":")[1]
        dxil_lines.append(list(filter(lambda x: len(x) > 0, map(lambda x: x.strip(', '), command.split(' ')))))
    return dxil_to_commands(input_lines, dxil_lines, inputs)

def process_inputs(inputs):
    res = dict()
    for inp in inputs:
        inp = inp.strip(' ').split()
        if inp[0] == "dcl_resource_texture2darray":
            res[inp[2]] = _Tex2DArray(inp[2], type_names.index(inp[1][1:].split(',')[0]))
        elif inp[0] == "dcl_resource_structured":
            res[inp[1]] = _SBuffer(inp[1], _UNKNOWN)
        elif inp[0] == "dcl_uav_raw":
            res[inp[1]] = _UAV(inp[1], _UNKNOWN)
        elif inp[0] == "dcl_uav_structured":
            res[inp[1]] = _UAVSBuffer(inp[1], _UNKNOWN)
    return res


def dxil_to_commands(input_lines, dxil_lines, external_inputs):
    processed = []
    context = _Context(external_inputs, process_inputs(input_lines))
    for command in dxil_lines:
        _squash_brackets_expression(command)
        # Classify command
        if command[0] in replace_table_operations:
            processed.append(_ResStatement(command[0], command[1], command[2:], context))
        elif command[0] in replace_table_operators:
            processed.append(_Statement(command[0], command[1:], context))
        elif command[0] in replace_table_special:
            processed.append(_StoreStatement(command[0], command[1:], context))
        else:
            raise Exception(f"Unsupported command {command}")
    for key, variables in context.replacer.items():
        names = {v.name for v in variables}
        processed = [_Initializer(variables[0], context)] + processed
        for p in processed:
            if isinstance(p, _ResStatement) and p.res.name in names:
                p.initialize = False
                p.res = variables[0]
    return "\n".join(list(map(str, processed)))
