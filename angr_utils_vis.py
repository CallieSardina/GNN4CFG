#! /usr/bin/env python

import angr
import os
from angrutils import plot_cfg, hook0, set_plot_style


def analyze(b, addr, name=None):
    start_state = b.factory.blank_state(addr=addr)
    start_state.stack_push(0x0)
    with hook0(b):
        cfg = b.analyses.CFGEmulated(fail_fast=True, starts=[addr], initial_state=start_state,
                                     context_sensitivity_level=2, keep_state=True, call_depth=100, normalize=True)
    for addr, func in proj.kb.functions.items():
        if func.name in ['main', 'verify']:
            plot_cfg(cfg, "./angr_cfgs/%s_%s_cfg" % (name, func.name), asminst=True, vexinst=False,
                     func_addr={addr: True}, debug_info=False, remove_imports=True, remove_path_terminator=True)

    # plot_cfg(cfg, "./angr_cfgs/%s_cfg_full" % (name), asminst=True, vexinst=True, debug_info=True, remove_imports=False, remove_path_terminator=False)
    #
    # plot_cfg(cfg, "./angr_cfgs/%s_cfg_classic" % (name), asminst=True, vexinst=False, debug_info=False, remove_imports=True, remove_path_terminator=True)
    plot_cfg(cfg, "./angr_cfgs/%s_cfg_classic" % (name), asminst=True, vexinst=False, debug_info=False,
             remove_imports=True, remove_path_terminator=True, format="raw")

    # for style in ['thick', 'dark', 'light', 'black', 'kyle']:
    #     set_plot_style(style)
    #     plot_cfg(cfg, "./angr_cfgs/%s_cfg_%s" % (name, style), asminst=True, vexinst=False, debug_info=False, remove_imports=True, remove_path_terminator=True)


def create_secure_cfgs():
    file_path = "bin_binaries_list.txt"
    with open(file_path, 'r') as file:
        for line in file:
            try:
                l = line.strip()[1:]
                print(l)
                n = l[5:]
                print(n)
                proj = angr.Project(l, load_options={'auto_load_libs': False})
                main = proj.loader.main_object.get_symbol("__libc_start_main")
                analyze(proj, main.rebased_addr, n)
            except Exception:
                print(Exception.__cause__)


def create_insecure_cfgs():
    folder_path = './insecure_binaries'

    # Loop through each file in the folder
    for filename in os.listdir(folder_path):
        # Construct the full path to the file
        file_path = os.path.join(folder_path, filename)
        # Check if the path is a file (not a subdirectory)
        if os.path.isfile(file_path):
            print(folder_path)
            print(filename)


if __name__ == "__main__":
    create_insecure_cfgs()
