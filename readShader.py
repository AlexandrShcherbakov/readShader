import collections

def _squash_brackets_expression(tokens):
    i = len(tokens) - 1
    while i >= 0:
        while "(" in tokens[i] and ")" not in tokens[i]:
            tokens[i] += ',' + tokens[i + 1]
            del tokens[i + 1]
        i -= 1

_Token = collections.namedtuple("Token", ["var", "comp"])

def _uncombine_var_and_comp(tokens):
    res_token_comp = tokens[1].split('.')[1]
    for i in range(2, len(tokens)):
        spl = tokens[i].split('.')
        if len(spl) != 2:
            continue
        tokens[i] = _Token(spl[0], spl[1])
        tokens[i] = spl[0] + "." + spl[1]
        if len(res_token_comp) == 1:
            continue
        comp = ""
        comps = {"x": 0, "y": 1, "z": 2, "w": 3}
        for c in res_token_comp:
            comp += spl[1][comps[c]]
        tokens[i] = _Token(spl[0], comp)
        tokens[i] = spl[0] + "." + comp


def _process_const(tokens):
    for i in range(len(tokens)):
        if not tokens[i].startswith("l("):
            continue
        components = tokens[i].count(",") + 1
        if components == 1:
            tokens[i] = tokens[i][2:-1]
        else:
            tokens[i] = "float" + str(components) + tokens[i][1:]


def replace_dxil_tokens_with_hlsl(dxil):
    replace_table_operations = {
        "utof": "asfloat({})",
        "mad": "{} * {} + {}",
        "ftou": "asuint({})",
        "mov": "{}",
        "ld_indexable(texture2darray)(float,float,float,float)": "{1}[{0}]",
        "ftoi": "asint({})",
        "ine": "{} != {}",
        "mul": "{} * {}",
        "ge": "{} >= {}",
        "movc": "{} ? {} : {}",
        "div": "{} / {}",
        "frc": "frac({})",
        "ld_structured_indexable(structured_buffer,stride=16)(mixed,mixed,mixed,mixed)": "{2}[{0} + {1}]",
        "and": "{} & {}",
        "add": "{} + {}",
        "dp3": "dot({}, {})",
        "sqrt": "sqrt({})",
        "min": "min({}, {})",
    }
    replace_table_special = {
        "atomic_iadd": "InterlockedAdd({}[{}], {})",
        "store_structured": "{}[{} + {}] = {}"
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
    for line in dxil.split('\n'):
        if not line.split(":")[0].strip().isdigit():
            hlsl += line + '\n'
            continue
        command = line.split(":")[1]
        tokens = list(filter(lambda x: len(x) > 0, map(lambda x: x.strip(', '), command.split(' '))))
        _squash_brackets_expression(tokens)
        _process_const(tokens)
        loc_tabs = tabs
        if tokens[0] in local_tab_modifiers:
            loc_tabs += local_tab_modifiers[tokens[0]]
        if tokens[0] in replace_table_operations:
            _uncombine_var_and_comp(tokens)
            hlsl += "  " * loc_tabs + tokens[1] + " = " + replace_table_operations[tokens[0]].format(*tokens[2:]) + ";\n"
        elif tokens[0] in replace_table_operators:
            hlsl += "  " * loc_tabs + replace_table_operators[tokens[0]].format(*tokens[1:]) + "\n"
        elif tokens[0] in replace_table_special:
            hlsl += "  " * loc_tabs + replace_table_special[tokens[0]].format(*tokens[1:]) + ";\n"
        else:
            print("Token {} should be implemented".format(tokens[0]))
        if tokens[0] in tabs_modifiers:
            tabs +=  tabs_modifiers[tokens[0]]

    return hlsl

