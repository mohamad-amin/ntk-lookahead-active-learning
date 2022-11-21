import torch
from itertools import starmap

__TYPE_TO_OPERATION = {
    'linear.weight': 'transpose',
    'linear.bias': 'eye',
    'conv.weight': 'conv',
    'conv.bias': 'eye',
    'bn.weight': 'eye',
    'bn.bias': 'eye',
}


def get_dict(dict_like):
    if isinstance(dict_like, dict):
        return dict_like
    else:
        return dict_like._dict


def _print_with_indent(msg, indent):
    print(''.join(['|  ']*(indent-1)) + '|--' + msg)


def _set_dict_value_nested(d, key, val):
    keys = key.split('.')
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = val


def _get_dict_value_nested(d, key):
    keys = key.split('.')
    for k in keys:
        d = d[k]
    return d


def _recursive_dict_keys(d, indent=0):
    for key in d.keys():
        _print_with_indent(key, indent)
        if isinstance(d[key], dict):
            _recursive_dict_keys(d[key], indent+1)


def _flatten_dict_with_names(d, name=''):
    names = []
    values = []
    for key in d.keys():
        if isinstance(d[key], dict):
            child_names, child_values = _flatten_dict_with_names(d[key], name=name+str(key)+'.')
            names.extend(child_names)
            values.extend(child_values)
        else:
            names.append(name + str(key))
            values.append(d[key])
    return names, values


def _detect_parameter_type(t_module, t_name, f_name, mat):

    module_name = t_module.__class__.__name__

    if module_name.find('Linear') != -1:
        return 'linear.weight' if 'weight' in t_name else 'linear.bias'
    if module_name.find('Conv') != -1:
        return 'conv.weight' if 'weight' in t_name else 'conv.bias'
    if module_name.find('BatchNorm') != -1:
        return 'bn.weight' if 'weight' in t_name else 'bn.bias'

    exit(f'Module not detected! Module: {t_module}, T Name: {t_name}, F name: {f_name}, mat shape: {mat.shape}')


def _convert_parameter_to_flax(t_parameter, type):
    op = __TYPE_TO_OPERATION[type]
    t_parameter = t_parameter.detach().cpu().numpy()
    if op == 'conv':
        return t_parameter.transpose(2, 3, 1, 0)
    elif op == 'transpose':
        return t_parameter.T
    elif op == 'eye':
        return t_parameter


def _convert_parameter_to_torch(f_parameter, type):
    op = __TYPE_TO_OPERATION[type]
    t_parameter = torch.from_numpy(f_parameter)
    if op == 'conv':
        return t_parameter.transpose(3, 2, 0, 1)
    elif op == 'transpose':
        return t_parameter.T
    elif op == 'eye':
        return t_parameter


def transfer_params_from_torch_model(model, params, dtype):

    model.eval()
    t_named_params = list(model.named_parameters())
    t_modules = list(model.modules())
    t_names = list(map(lambda x: x[0], t_named_params))
    t_params = list(map(lambda x: x[1], t_named_params))

    params_dict = get_dict(params['params'])
    f_names, f_params = _flatten_dict_with_names(params_dict)

    t_modules = list(filter(lambda x: hasattr(x, 'weight'), t_modules))
    t_modules = [item for item in t_modules for i in (range(2) if item.bias is not None else range(1))]

    types = list(starmap(_detect_parameter_type, zip(t_modules, t_names, f_names, t_params)))

    for i in range(len(f_params)):
        try:
            f_params[i] = _convert_parameter_to_flax(t_params[i], types[i]).astype(dtype)
        except:
            import IPython; IPython.embed()

    for i in range(len(f_params)):
        name = f_names[i]
        _set_dict_value_nested(params_dict, name, f_params[i])

    flax_params = {'params': params_dict}

    if 'batch_stats' in params.keys():

        batch_stats_dict = get_dict(params['batch_stats'])
        batch_indices = [i for i, type in enumerate(types) if 'bn' in type]
        f_b_names, f_b_params = _flatten_dict_with_names(batch_stats_dict)

        b_i = 0
        for i in batch_indices:
            name_hierarchy = t_names[i].split('.')
            p = model
            for n in name_hierarchy[:-1]:
                p = getattr(p, n)
            attr = 'running_mean' if f_b_names[b_i].endswith('mean') else 'running_var'
            f_b_params[b_i] = getattr(p, attr).detach().cpu().numpy().astype(dtype)
            b_i += 1

        for b_i in range(len(f_b_params)):
            name = f_b_names[b_i]
            _set_dict_value_nested(batch_stats_dict, name, f_b_params[b_i])

        flax_params['batch_stats'] = batch_stats_dict

    return flax_params


def transfer_params_from_flax_model(model, params, dtype):

    model.eval()
    t_named_params = list(model.named_parameters())
    t_modules = list(model.modules())
    t_names = list(map(lambda x: x[0], t_named_params))
    t_params = list(map(lambda x: x[1], t_named_params))

    params_dict = get_dict(params['params'])
    f_names, f_params = _flatten_dict_with_names(params_dict)

    t_modules = list(filter(lambda x: hasattr(x, 'weight'), t_modules))
    t_modules = [item for item in t_modules for i in range(2)]

    types = list(starmap(_detect_parameter_type, zip(t_modules, t_names, f_names, t_params)))
    for i in range(len(f_params)):
        try:
            t_params[i] = _convert_parameter_to_torch(f_params[i], types[i]).astype(dtype)
        except:
            import IPython; IPython.embed()

    # this part sounds tricky
    if 'batch_stats' in params.keys():

        batch_stats_dict = get_dict(params['batch_stats'])
        batch_indices = [i for i, type in enumerate(types) if 'bn' in type]
        f_b_names, f_b_params = _flatten_dict_with_names(batch_stats_dict)

        b_i = 0
        for i in batch_indices:
            name_hierarchy = t_names[i].split('.')
            p = model
            for n in name_hierarchy[:-1]:
                p = getattr(p, n)
            attr = 'running_mean' if f_b_names[b_i].endswith('mean') else 'running_var'
            f_b_params[b_i] = getattr(p, attr).detach().cpu().numpy().astype(dtype)
            b_i += 1

        for b_i in range(len(f_b_params)):
            name = f_b_names[b_i]
            _set_dict_value_nested(batch_stats_dict, name, f_b_params[b_i])

    return t_params