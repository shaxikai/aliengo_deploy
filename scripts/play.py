'''
**************************************************************************

* @file         play.py
* @author       Wei Wang -> shaxikai@outlook.com
* @date         2025.4.1
* @version      V1.0.0"
* @brief        run policy

"*************************************************************************
'''

import os
from utils.him_deploy import System

if __name__ == "__main__":

    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    cfg_pn = current_dir + "/../config/aliengo.yaml"
    system = System(cfg_pn)
    system.run()