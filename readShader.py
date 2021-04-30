import collections

def _squash_brackets_expression(tokens):
    i = len(tokens) - 1
    while i >= 0:
        while "(" in tokens[i] and ")" not in tokens[i]:
            tokens[i] += ',' + tokens[i + 1]
            del tokens[i + 1]
        i -= 1

class _Token:
    def __init__(self, var, comp):
        self.var = var
        self.comp = comp

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
        spl = tokens[i].rsplit('.', 1)
        if len(spl) != 2:
            continue
        tokens[i] = _Token(spl[0], spl[1])
        if len(spl[1]) == 1 or i == 1:
            continue
        comp = ""
        comps = {"x": 0, "y": 1, "z": 2, "w": 3}
        for c in res_token_comp[i]:
            comp += spl[1][comps[c]]
        tokens[i] = _Token(spl[0], comp)


def _process_const(tokens):
    for i in range(len(tokens)):
        if not tokens[i].startswith("l("):
            continue
        components = tokens[i].count(",") + 1
        if components == 1:
            tokens[i] = tokens[i][2:-1]
        else:
            tokens[i] = "float" + str(components) + tokens[i][1:] + ".xyzw"[:components + 1]


class _Variable:
    def __init__(self, name, reg, comp):
        self.name = name
        self.reg = reg
        self.comp = comp

    def __str__(self):
        return self.name
    
    def gen_reg_to_vars_pairs(self):
        common_comps = "xyzw"[:len(self.comp)]
        return {_Token(self.reg, self.comp[i]) : _Token(self.name, common_comps[i]) for i in range(len(common_comps))}

def _token_to_single_regs(token):
    return [_Token(token.var, i) for i in token.comp]

def _replace_reg_with_var(reg, variables_map):
    for variable in variables_map.values():
        if reg.var == variable.reg and reg.comp in variable.comp:
            return _Token(variable.name, "xyzw"[variable.comp.index(reg.comp)])

def _is_reg(token):
    return isinstance(token, _Token) and token.var[0] == "r" and token.var[1:].isdigit()

def _single_vars_to_vec(vars):
    if len(vars) == 1:
        return vars[0]
    return f"float{len(vars)}({', '.join(list(map(str, vars)))})"

def _find_var_in_scopes(scopes, reg):
    for i in range(len(scopes) - 1, -1, -1):
        if reg in scopes[i]:
            return scopes[i][reg]

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
            var = _Variable(variable_name, tokens[1].var, tokens[1].comp)
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
            hlsl += "  " * loc_tabs + str(tokens[1]) + " = " + replace_table_operations[tokens[0]].format(*tokens[2:]) + ";\n"
        elif tokens[0] in replace_table_operators:
            hlsl += "  " * loc_tabs + replace_table_operators[tokens[0]].format(*tokens[1:]) + "\n"
        elif tokens[0] in replace_table_special:
            hlsl += "  " * loc_tabs + replace_table_special[tokens[0]].format(*tokens[1:]) + ";\n"
        else:
            print("Token {} should be implemented".format(tokens[0]))
        if tokens[0] in tabs_modifiers:
            tabs +=  tabs_modifiers[tokens[0]]

    return hlsl

