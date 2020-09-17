from classes.Model import Model
from math import floor

model_name = "model_char_short_5"
songname = "text_out.txt"

model = Model()
model.load(model_name)


print("GENERATING: ", songname)
model.generateFreewheel_text(songname, 10, 1000)

print()
print("SAVING embed data")
#model.embedInfoToFile("", filename=model_name+"_info.txt")