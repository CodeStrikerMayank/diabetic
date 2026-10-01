from project1 import file_management , artifical_neuron
import numpy as np
import pandas as pd
from rich.console import Console 
from rich.panel import Panel 
from rich.live import Live
import time
import threading as td 



print("\033c")
console = Console()
frame = ["Laodiong","Loading..","Loading...","Loading...."]
with Live(None,refresh_per_second=4,console=console)as live:
    for i in frame:
        live.update(Panel(i,style="bold green"))
        time.sleep(1)


class est(file_management):

    def __init__(self):
        super().__init__()
        print(self.y.shape)
        print(self.x.shape)

class test(artifical_neuron):

    def __init__(self):
        super().__init__()
        self.neuron_brain_function()
        

