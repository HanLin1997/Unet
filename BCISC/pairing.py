import os

dictionary = {"baso": 0, "eosi": 1, "lymp": 2, "mono": 3, "neut": 4}

def ori_pairing(folder_path, pairing_text, label):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.jpg'):
                file_path = os.path.join(root, file)
                mask_path = os.path.splitext(file_path)[0] + ".png"
                if not os.path.exists(mask_path):
                    mask_path = os.path.splitext(file_path)[0] + "(1).png"
                if os.path.exists(mask_path):
                    pairing_text = pairing_text + f"{file_path},{mask_path},{dictionary[label]}\n"
                else:
                    raise FileNotFoundError(f'Path "{file_path}" does not exist.')
    return pairing_text

def sub_pairing(folder_path, pairing_text):
    for root, dirs, files in os.walk(folder_path + "/images"):
        for file in files:
            label = file.split('_')[0]
            file_path = os.path.join(root, file)
            mask_path = folder_path + "/masks/" + os.path.splitext(file)[0] + ".png"
            if os.path.exists(mask_path):
                pairing_text = pairing_text + f"{file_path},{mask_path},{dictionary[label]}\n"
            else:
                raise FileNotFoundError(f'Path "{mask_path}" does not exist.')
    return pairing_text

if __name__ == "__main__":
    pairing_text = ""
    pairing_text = ori_pairing("BCISC/original-images/baso", pairing_text, "baso")
    pairing_text = ori_pairing("BCISC/original-images/eosi", pairing_text, "eosi")
    pairing_text = ori_pairing("BCISC/original-images/lymp", pairing_text, "lymp")
    pairing_text = ori_pairing("BCISC/original-images/mono", pairing_text, "mono")
    pairing_text = ori_pairing("BCISC/original-images/neut", pairing_text, "neut")
    pairing_text = sub_pairing("BCISC/sub-images", pairing_text)
    with open("./train.txt", 'w') as f:
        f.write(pairing_text)
