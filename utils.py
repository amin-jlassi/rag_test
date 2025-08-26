def exist(file_name) : 
    with open("pdf_list.txt" , "r") as f : 
        l = f.readline()
        while l != "" : 
            if file_name == l[:len(l)-1] : 
                return True
            l = f.readline()
    return False

def savePdf_inFile(file_name) : 
    with open("pdf_list.txt" , "a") as f : 
        f.write(file_name+"\n")
        print("pdf added to pdf_list.txt")