from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import codecs
import yaml
import six
import time

from ast import literal_eval

class SegmentronConfig(dict):
    def __init__(self, *args, **kwargs):
        super(SegmentronConfig, self).__init__(*args, **kwargs)
        self.immutable = False

    def __setattr__(self, key, value, create_if_not_exist=True):
        if key in ["immutable"]:
            self.__dict__[key] = value
            return

        t = self
        keylist = key.split(".")
        for k in keylist[:-1]:
            t = t.__getattr__(k, create_if_not_exist)

        t.__getattr__(keylist[-1], create_if_not_exist)
        t[keylist[-1]] = value

    def __getattr__(self, key, create_if_not_exist=True):
        if key in ["immutable"]:
            if key not in self.__dict__:
                self.__dict__[key] = False
            return self.__dict__[key]

        if not key in self:
            if not create_if_not_exist:
                raise KeyError
            self[key] = SegmentronConfig()
        return self[key]

    def __setitem__(self, key, value):
        #
        if self.immutable:
            raise AttributeError(
                'Attempted to set "{}" to "{}", but SegConfig is immutable'.
                format(key, value))
        #
        if isinstance(value, six.string_types):
            try:
                value = literal_eval(value)
            except ValueError:
                pass
            except SyntaxError:
                pass
        super(SegmentronConfig, self).__setitem__(key, value)

    def update_from_other_cfg(self, other):
        if isinstance(other, dict):
            other = SegmentronConfig(other)
        assert isinstance(other, SegmentronConfig)
        cfg_list = [("", other)]
        while len(cfg_list):
            prefix, tdic = cfg_list[0]
            cfg_list = cfg_list[1:]
            for key, value in tdic.items():
                key = "{}.{}".format(prefix, key) if prefix else key
                if isinstance(value, dict):
                    cfg_list.append((key, value))
                    continue
                try:
                    self.__setattr__(key, value, create_if_not_exist=False)
                except KeyError:
                    raise KeyError('Non-existent config key: {}'.format(key))

    def remove_irrelevant_cfg(self):
        model_name = self.MODEL.MODEL_NAME

        from ..models.model_zoo import MODEL_REGISTRY
        model_list = MODEL_REGISTRY.get_list()
        model_list_lower = [x.lower() for x in model_list]
        
        assert model_name.lower() in model_list_lower, "Expected model name in {}, but received {}"\
            .format(model_list, model_name)
        pop_keys = []
        for key in self.MODEL.keys():
            if key.lower() in model_list_lower:
                if model_name.lower() == 'pointrend' and \
                    key.lower() == self.MODEL.POINTREND.BASEMODEL.lower():
                    continue
            if key.lower() in model_list_lower and key.lower() != model_name.lower():
                pop_keys.append(key)
        for key in pop_keys:
            self.MODEL.pop(key)



    def check_and_freeze(self):
        self.TIME_STAMP = time.strftime('%Y-%m-%d-%H-%M', time.localtime())
        # TODO: remove irrelevant config and then freeze
        self.remove_irrelevant_cfg()
        self.immutable = True

    def update_from_list(self, config_list):
        if len(config_list) % 2 != 0:
            raise ValueError(
                "Command line options config format error! Please check it: {}".
                format(config_list))
        for key, value in zip(config_list[0::2], config_list[1::2]):
            try:
                self.__setattr__(key, value, create_if_not_exist=False)
            except KeyError:
                raise KeyError('Non-existent config key: {}'.format(key))

    def update_from_file(self, config_file):
        with codecs.open(config_file, 'r', 'utf-8') as file:
            loaded_cfg = yaml.load(file, Loader=yaml.FullLoader)
        self.update_from_other_cfg(loaded_cfg)

    def set_immutable(self, immutable):
        self.immutable = immutable
        for value in self.values():
            if isinstance(value, SegmentronConfig):
                value.set_immutable(immutable)

    def is_immutable(self):
        return self.immutable