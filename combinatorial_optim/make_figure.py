import numpy as np
import matplotlib.pyplot as plt
#from matplotlib.font_manager import FontProperties 
#font_set = FontProperties(fname=r"C:\Windows\Fonts\simsunb.ttf", size=12)
a = [0.279,0.56,0.626,0.645,0.667,0.67,0.695,0.705,0.665,0.638,0.65,0.628,0.625,0.623,0.613,
    0.591,0.602,0.57,0.574,0.549,0.533,0.522,0.527,0.517,0.507,0.495,0.5
]
a = [0.279,	0.56,0.626,0.645,0.667,0.67,0.695,0.705,
0.768,0.7344,0.789,0.793,0.777,0.798,0.777,0.801,0.821,0.771,0.786,0.783,0.784,0.805,0.778,0.762,0.782,0.791,0.79
]
x = range(1,len(a)+1)
plt.plot(x,a)
#plt.title("",fontproperties=font_set)
plt.xlabel("camera number")
plt.ylabel("mAP")
#plt.grid()
plt.show()