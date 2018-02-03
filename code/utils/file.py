def loadtxt(filepath, flat = True):
    data = []
    lol = 1
    with open(filepath, "r") as datafile:
        for line in datafile:
            if line.startswith("#"):
                continue
            slce = map(lambda x: float(x.strip()), line.rstrip().split(" "))
            if flat:
                data.extend(slce)
            else:
                data.append(slce)
    return data


def unzip(zipped_list):
    #https://stackoverflow.com/questions/13635032/what-is-the-inverse-function-of-zip-in-python#13635074
    return izip(*zipped_list)