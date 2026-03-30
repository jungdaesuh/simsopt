def make_module_jit_hooks(parameters, value=False):
    original_jit = parameters["jit"]

    def set_up_module():
        parameters["jit"] = value

    def tear_down_module():
        parameters["jit"] = original_jit

    return set_up_module, tear_down_module
