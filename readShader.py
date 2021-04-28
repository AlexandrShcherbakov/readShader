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
        if tokens[0].startswith("ld_structured_indexable"):
            tokens[0] += "," + tokens[1]
            del tokens[1]
        loc_tabs = tabs
        if tokens[0] in local_tab_modifiers:
            loc_tabs += local_tab_modifiers[tokens[0]]
        if tokens[0] in replace_table_operations:
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

