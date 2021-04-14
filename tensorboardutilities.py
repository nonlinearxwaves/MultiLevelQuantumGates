# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 14:47:38 2018

@author: nonli
"""

class tensorboardutilities:
    def getdirname():
        # return the tensorboard dir depending on the machine name and time
        import socket
        from datetime import datetime
        now = datetime.now()    
        if socket.gethostname()=='hawaii':
            tensorboarddir = "/home/claudio/tensorflow_claudio_logs/" + now.strftime("%Y%m%d-%H%M%S") + "/"
        elif socket.gethostname()=='DESKTOP-667QSME':
            tensorboarddir = "c:/Users/claudio/tensorflow_claudio_logs/" + now.strftime("%Y%m%d-%H%M%S") + "/"    
        else:
            tensorboarddir = "c:/Users/nonli/tensorflow_claudio_logs/" + now.strftime("%Y%m%d-%H%M%S") + "/"
        return tensorboarddir
                