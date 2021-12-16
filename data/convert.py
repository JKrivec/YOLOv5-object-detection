import sys, os

# Get src dir
src_dir = sys.argv[1]


# Get current path
curr_path =  os.path.dirname(os.path.realpath(__file__))
# Get FILES ONLY current dir (dot traverse directories)
file_names = next(os.walk(curr_path + "/" + src_dir), (None, None, []))[2]  # [] if no file

# Create a converted_ dir
converted_path = "converted_" + src_dir
if not os.path.exists(converted_path):
	os.mkdir(converted_path)

# Transform each line in each file and write the file into converted_dir
for file_name in file_names:
	print(file_name)
	file = open(curr_path + '\\' + src_dir + '\\' +file_name, "r")
	file_dest = open(curr_path + '\\' + converted_path + '\\' +file_name, "w")
	for line in file:
		split_line = line.split(" ")
		print(split_line)
		dw = 1./480
		dh = 1./360
		x = float(split_line[1])
		y = float(split_line[2])
		w = float(split_line[3])
		h = float(split_line[4])
		x = (x+(w/2))*dw
		w = w*dw
		y = (y+(h/2))*dh
		h = h*dh

		print(str(split_line[0]) + " " + str(round(x, 6)) + " " + str(round(y, 6)) + " " + str(round(w, 6)) + " " + str(round(h, 6)) + "\n")
		file_dest.write(str(split_line[0]) + " " + str(round(x, 6)) + " " + str(round(y, 6)) + " " + str(round(w, 6)) + " " + str(round(h, 6)) + "\n")
	
	file.close()
	file_dest.close()
