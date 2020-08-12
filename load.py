from classes.Model import Model

model_name = "model_1"
songname = model_name+"_out.wav"

model = Model()
model.load(model_name)


print("GENERATING: ", songname)
model.generateSong(songname, 450, 3, 22050)

print()
print("SAVING embed data")
#model.embedInfoToFile("", filename=model_name+"_info.txt")