import collections
import inspect
import json
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from dacite import from_dict

from detr.loggers import log
from detr.utils.files import load_yaml, relpath
from detr.utils.registry import Registry


@dataclass
class AbstractConfig:
    def to_dict(self) -> dict:
        dct = {}
        for field in fields(self):
            field_name = field.name
            field_value = getattr(self, field_name)
            if hasattr(field_value, "to_dict"):
                dct[field_name] = field_value.to_dict()
            elif isinstance(field_value, dict):  # config class nested in dict (e.g. optimizers)
                init_dict = {}
                for k, v in field_value.items():
                    if hasattr(v, "to_dict"):
                        init_dict[k] = v.to_dict()
                    else:
                        init_dict[k] = v
                dct[field_name] = init_dict
            elif isinstance(field_value, list):  # handle lists of custom formats (e.g. loss)
                dct[field_name] = [item.to_dict() if hasattr(item, "to_dict") else item for item in field_value]
            else:
                dct[field_name] = field_value
        return dct

    @classmethod
    def from_dict(cls, cfg_dict: dict):
        return from_dict(data_class=cls, data=cfg_dict)

    @classmethod
    def from_yaml_to_dict(cls, filepath: str | Path) -> dict:
        cfg_dict = load_yaml(filepath)
        cfg_dict = update_config(cfg_dict, cls)
        log.info(f"Loaded config from '{relpath(filepath)}'")
        return cfg_dict

    @classmethod
    def from_yaml(cls, filepath: str | Path):
        cfg_dict = cls.from_yaml_to_dict(filepath)
        return cls.from_dict(cfg_dict)


def get_default_kwargs(cls) -> dict[str, dict[str, Any]]:
    default_kwargs = {}

    for method_name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
        signature = inspect.signature(method)
        kwargs = {
            name: param.default
            for name, param in signature.parameters.items()
            if param.default is not inspect.Parameter.empty
        }
        if kwargs:
            default_kwargs[method_name] = kwargs

    return default_kwargs


def parse_list_dct_cfgs(cfgs: list[dict], registry: Registry) -> list[Any]:
    objects = []
    for cfg in cfgs:
        for name, params in cfg.items():
            cls_name = registry[name]
            if params is None:
                params = {}
            obj = cls_name(**params)
            default_kwargs = get_default_kwargs(cls_name).get("__init__", {})
            kwargs_info = ", ".join(f"{k}={v}" for k, v in params.items())
            for k in params:
                if k in default_kwargs:
                    default_kwargs.pop(k)
            if len(default_kwargs) > 0:
                kwargs_info += ", " + ", ".join(f"{k}={v}" for k, v in default_kwargs.items())
            log.info(f"\t- {obj.__class__.__name__}({kwargs_info})")
            objects.append(obj)
    return objects


def update_config(cfg_dict: dict, config_cls: AbstractConfig) -> dict:
    update_dct = parse_args_for_config(config_cls)
    if len(update_dct) > 0:
        msg = (
            f"Updating config dict using CLI args:\n{json.dumps(update_dct, indent=4)}\n" f"Config updates summary: \n"
        )
        cfg_dict, msg = update_dict(cfg_dict, update_dct, msg)
        log.info(msg)
    return cfg_dict


def parse_args_for_config(config_cls: AbstractConfig = AbstractConfig) -> dict:
    def parse_dict(update_dct: dict) -> dict:
        output_dict = {}
        for key, value in update_dct.items():
            parts = key.split(".")
            current_dict = output_dict
            for part in parts[:-1]:
                if part not in current_dict:
                    current_dict[part] = {}
                current_dict = current_dict[part]
            current_dict[parts[-1]] = value
        return output_dict

    args = sys.argv
    valid_args = [arg[2:] for arg in args if arg[:2] == "--" and "=" in arg]
    name_values = [arg.split("=") for arg in valid_args]
    name2value = {name: value for name, value in name_values}
    name2value = parse_dict(name2value)
    cfg_fields = fields(config_cls)
    cfg_fields_names = [field.name for field in cfg_fields]
    name2value = {name: value for name, value in name2value.items() if name in cfg_fields_names}
    return name2value


def parse_cli_value(value: str) -> int | float | str | None:
    if value in ["None", "none", "null"]:
        return None
    elif "." in value:
        try:
            return float(value)
        except ValueError:
            return value
    else:
        if value in ["true", "True"]:
            return True
        elif value in ["false", "False"]:
            return False
        try:
            return int(value)
        except ValueError:
            return value


def update_dict(dct: dict, update_dct: dict, msg: str = "") -> tuple[dict, str]:
    for k, v in update_dct.items():
        if isinstance(v, collections.abc.Mapping):
            dct[k], msg = update_dict(dct.get(k, {}), v, msg)
        else:
            new_value = parse_cli_value(v)
            msg += f"\t- `{k}` value updated from `{dct[k]}` to `{new_value}`\n"
            dct[k] = new_value
    return dct, msg
