import collections

type_names = ["UNKNOWN", "int", "uint", "float"]
_FLOAT = 3
_UINT = 2
_INT = 1
_UNKNOWN = 0

def _squash_brackets_expression(tokens):
    i = len(tokens) - 1
    while i >= 0:
        while "(" in tokens[i] and ")" not in tokens[i]:
            tokens[i] += ',' + tokens[i + 1]
            del tokens[i + 1]
        i -= 1

class _Token:
    def __init__(self, var, comp, tp):
        self.var = var
        self.comp = comp
        self.type = tp

    def __str__(self):
        return self.var + "." + self.comp

    def __eq__(self, another):
        return self.var == another.var and self.comp == another.comp

    def __hash__(self):
        return hash(f"{self.var}.{self.comp}")

def _uncombine_var_and_comp(tokens):
    if '.' not in tokens[1]:
        return
    res_token_comp = tokens[1].split('.')[1]
    if tokens[0].startswith("dp"):
        res_token_comp = "xyzw"[:int(tokens[0][2])]
    res_token_comp = [res_token_comp for _ in range(len(tokens))]
    if "ld_indexable(texture2darray)" in tokens[0]:
        res_token_comp[2] = "xyz"
    for i in range(1, len(tokens)):
        tp = _UNKNOWN
        spl = tokens[i].rsplit('.', 1)
        if spl[0] == "vThreadID":
            tp = _UINT
        if len(spl) != 2:
            continue
        if all(map(lambda x: x.isdigit(), spl)):
            tp = _FLOAT
        if spl[0].startswith('uint'):
            tp = _UINT
        elif spl[0].startswith('int'):
            tp = _INT
        elif spl[0].startswith('float'):
            tp = _FLOAT
        tokens[i] = _Token(spl[0], spl[1], tp)
        if len(spl[1]) == 1 or i == 1:
            continue
        comp = ""
        comps = {"x": 0, "y": 1, "z": 2, "w": 3}
        for c in res_token_comp[i]:
            comp += spl[1][comps[c]]
        tokens[i] = _Token(spl[0], comp, tp)


def _process_const(tokens):
    for i in range(len(tokens)):
        if not tokens[i].startswith("l("):
            continue
        components = tokens[i].count(",") + 1
        if components == 1:
            tokens[i] = tokens[i][2:-1]
        else:
            tp = _FLOAT if '.' in tokens[i][1:] else _INT
            tokens[i] = type_names[tp] + str(components) + tokens[i][1:] + ".xyzw"[:components + 1]

class _Variable:
    def __init__(self, name, reg, comp, _type):
        self.name = name
        self.reg = reg
        self.comp = comp
        self.type = _type

    def __str__(self):
        return self.name
    
    def gen_reg_to_vars_pairs(self):
        common_comps = "xyzw"[:len(self.comp)]
        return {_Token(self.reg, self.comp[i], _UNKNOWN) : _Token(self.name, common_comps[i], self.type) for i in range(len(common_comps))}

    def gen_type(self):
        res = type_names[self.type]
        if len(self.comp) > 1:
            res += str(len(self.comp))
        return res

def _token_to_single_regs(token):
    return [_Token(token.var, i, token.type) for i in token.comp]

def _replace_reg_with_var(reg, variables_map):
    for variable in variables_map.values():
        if reg.var == variable.reg and reg.comp in variable.comp:
            return _Token(variable.name, "xyzw"[variable.comp.index(reg.comp)])

def _is_reg(token):
    return isinstance(token, _Token) and token.var[0] == "r" and token.var[1:].isdigit()

def _default_type_extr(tokens):
        tp = _UNKNOWN
        for t in tokens:
            if isinstance(t, _Token):
                tp = max(tp, t.type)
            elif isinstance(t, str):
                if t.isdigit():
                    tp = max(tp, _INT)
        return tp

def _single_vars_to_vec(vars):
    if len(vars) == 1:
        return vars[0]
    tp = _default_type_extr(vars)
    name = f"{type_names[tp]}{len(vars)}({', '.join(list(map(str, vars)))})"
    return _Token(name, "xyzw"[:len(vars)], tp)

def _find_var_in_scopes(scopes, reg):
    for i in range(len(scopes) - 1, -1, -1):
        if reg in scopes[i]:
            return scopes[i][reg]
    raise Exception("Unknown variable")

def replace_dxil_tokens_with_hlsl(dxil):
    replace_table_operations = {
        "utof": "asfloat({})",
        "mad": "{} * {} + {}",
        "ftou": "asuint({})",
        "mov": "{}",
        "ld_indexable(texture2darray)(float,float,float,float)": "{1.var}[{0}].{1.comp}",
        "ftoi": "asint({})",
        "ine": "{} != {}",
        "mul": "{} * {}",
        "ge": "{} >= {}",
        "movc": "{} ? {} : {}",
        "div": "{} / {}",
        "frc": "frac({})",
        "ld_structured_indexable(structured_buffer,stride=16)(mixed,mixed,mixed,mixed)": "{2.var}[{0} + {1}].{2.comp}",
        "and": "{} & {}",
        "add": "{} + {}",
        "dp3": "dot({}, {})",
        "sqrt": "sqrt({})",
        "min": "min({}, {})",
    }
    replace_table_special = {
        "atomic_iadd": "InterlockedAdd({}[{}], {})",
        "store_structured": "{0.var}[{1} + {2}].{0.comp} = {3}"
    }
    replace_table_operators = {
        "if_nz": "if ({}) {{",
        "else": "}} else {{",
        "endif": "}}",
        "ret": ""
    }
    tabs_modifiers = {
        "if_nz": 1,
        "endif": -1
    }
    local_tab_modifiers = {
        "else": -1,
        "endif": -1
    }

    type_by_instr = collections.defaultdict(lambda: _default_type_extr)
    type_by_instr.update({
        "utof": lambda args: _FLOAT,
        "ftou": lambda args: _UINT,
        "ftoi": lambda args: _INT
    })

    hlsl = ""
    tabs = 0
    tokens_lines = []
    for line in dxil.split('\n'):
        if not line.split(":")[0].strip().isdigit():
            hlsl += line + '\n'
            continue
        command = line.split(":")[1]
        tokens = list(filter(lambda x: len(x) > 0, map(lambda x: x.strip(', '), command.split(' '))))
        _squash_brackets_expression(tokens)
        _process_const(tokens)
        if tokens[0] in replace_table_operations or tokens[0] in replace_table_special:
            _uncombine_var_and_comp(tokens)
        tokens_lines.append(tokens)

    variables_map = [dict()]
    variable_idx = 0
    for tokens in tokens_lines:
        if tokens[0] in replace_table_operations:
            for i in range(2, len(tokens)):
                if not _is_reg(tokens[i]):
                    continue
                single_regs = _token_to_single_regs(tokens[i])
                single_vars = [_find_var_in_scopes(variables_map, reg) for reg in single_regs]
                tokens[i] = _single_vars_to_vec(single_vars)

            variable_name = f"v_{variable_idx}"
            variable_idx += 1
            var = _Variable(variable_name, tokens[1].var, tokens[1].comp, type_by_instr[tokens[0]](tokens[2:]))
            variables_map[-1].update(var.gen_reg_to_vars_pairs())
            tokens[1] = var
        if tokens[0].startswith("if"):
            variables_map.append(dict())
        elif tokens[0] == "else":
            variables_map[-1] = dict()
        elif tokens[0] == "endif":
            variables_map = variables_map[:-1]
                

    for tokens in tokens_lines:
        loc_tabs = tabs
        if tokens[0] in local_tab_modifiers:
            loc_tabs += local_tab_modifiers[tokens[0]]
        if tokens[0] in replace_table_operations:
            hlsl += "  " * loc_tabs + tokens[1].gen_type() + " " + str(tokens[1]) + " = " + replace_table_operations[tokens[0]].format(*tokens[2:]) + ";\n"
        elif tokens[0] in replace_table_operators:
            hlsl += "  " * loc_tabs + replace_table_operators[tokens[0]].format(*tokens[1:]) + "\n"
        elif tokens[0] in replace_table_special:
            hlsl += "  " * loc_tabs + replace_table_special[tokens[0]].format(*tokens[1:]) + ";\n"
        else:
            print("Token {} should be implemented".format(tokens[0]))
        if tokens[0] in tabs_modifiers:
            tabs +=  tabs_modifiers[tokens[0]]

    return hlsl

