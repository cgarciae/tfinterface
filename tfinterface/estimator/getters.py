
import os
import sys
from copy import deepcopy
import time
import subprocess
import shutil


class FileGetter(object):

    @classmethod
    def get(cls, url, *args , **kwargs):
        
        rm = kwargs.pop("rm", False)

        hash_name = str(hash(url)).replace("-", "0")
        filename = os.path.basename(url)

        home_path = os.path.expanduser('~')
        model_dir_base = os.path.join(home_path, ".local", "tfinterface", "frozen_graphs", hash_name)
        model_path = os.path.join(model_dir_base, filename)

        if rm or (os.path.exists(model_dir_base) and len(os.listdir(model_dir_base)) == 0):
            shutil.rmtree(model_dir_base)

        if not os.path.exists(model_dir_base):
            os.makedirs(model_dir_base)

            cmd = "gsutil -m cp -R {source_folder} {dest_folder}".format(
                source_folder = url,
                dest_folder = model_dir_base,
            )

            print("CMD: {}".format(cmd))

            subprocess.check_call(
                cmd,
                stdout = subprocess.PIPE, shell = True,
            )

        return cls(model_path, *args, **kwargs)


class FolderGetter(object):

    @classmethod
    def get(cls, url, **kwargs):

        rm = kwargs.pop("rm", False)

        hash_name = str(hash(url)).replace("-", "0")

        home_path = os.path.expanduser('~')
        model_dir_base = os.path.join(home_path, ".local", "tfinterface", "saved_models", hash_name)

        if rm or (os.path.exists(model_dir_base) and len(os.listdir(model_dir_base)) == 0):
            shutil.rmtree(model_dir_base)

        if not os.path.exists(model_dir_base):
            os.makedirs(model_dir_base)

            cmd = "gsutil -m cp -R {source_folder}/* {dest_folder}".format(
                source_folder = url,
                dest_folder = model_dir_base,
            )

            print("CMD: {}".format(cmd))

            subprocess.check_call(
                cmd,
                stdout = subprocess.PIPE, shell = True,
            )

        return cls(model_dir_base, **kwargs)