'''
**************************************************************************

* @file         play.py
* @author       Wei Wang -> shaxikai@outlook.com
* @date         2025.4.1
* @version      V1.1.0"
* @brief        run policy

"*************************************************************************
'''

import os
from walk_these_ways.wtw_system import WTWSystem

if __name__ == "__main__":

    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    cfg_pn = current_dir + "/../walk_these_ways/config/aliengo.yaml"
    system = WTWSystem(cfg_pn)
    system.run()