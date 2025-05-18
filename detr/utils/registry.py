import inspect
from typing import Callable, Type


def get_class_location(cls) -> tuple[str, int]:
    file_path = inspect.getfile(cls)
    source_lines, starting_line = inspect.getsourcelines(cls)
    return file_path, starting_line


class Registry:
    def __init__(self):
        self.registry = {}

    def __setitem__(self, key: str, new_value: Type):
        if key in self.registry:
            from detr.loggers import log  # avoid circular import

            cls_value = self.registry[key]
            if cls_value == new_value:
                return
            old_fpath, old_line = get_class_location(cls_value)
            new_fpath, new_line = get_class_location(new_value)

            e = ValueError(
                f'Trying to register class from File "{new_fpath}", line {new_line}). \n'
                f"Class with name `{key}` is already registered. \n"
                f'Registered class value: {cls_value} (File "{old_fpath}", line {old_line}). \n'
            )
            log.exception(e)
            raise e
        self.registry[key] = new_value

    def __getitem__(self, key: str) -> Type:
        if key not in self.registry:
            from detr.loggers import log  # avoid circular import

            e = KeyError(
                f"There is no class with name `{key}` registered. \n"
                f"Registered class names: {list(self.registry.keys())}. \n"
                f"NOTE: Remember to import all decorated classes in __init__.py"
            )
            log.exception(e)
            raise e
        return self.registry[key]


def create_register_decorator(register: Registry) -> Callable:
    def register_decorator(cls: Type) -> Type:
        register[cls.__name__] = cls
        return cls

    return register_decorator
