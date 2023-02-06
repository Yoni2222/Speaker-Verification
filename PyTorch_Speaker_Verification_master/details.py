import os


class Details:
    def __init__(self, first_name=None, last_name=None, id=None, phone=None, address=None, city=None):
        self.details_dict = dict([])
        self.details_dict["first_name"] = first_name
        self.details_dict["last_name"] = last_name
        self.details_dict["id"] = id
        self.details_dict["phone"] = phone
        self.details_dict["address"] = address
        self.details_dict["city"] = city

    def __getitem__(self, item):
        return self.__getattr__(item)

    def __getattr__(self, item):
        if item not in self.details_dict.keys():
            raise Exception(str(item) + " - is not exist, the options are : " + str(list(self.details_dict.keys())))
        return self.details_dict[item]

    def __str__(self):
        string = "Details object:"

        for title in self.details_dict.keys():
            string += "\n" + title + " : " + self.details_dict[title]
        return string

    def to_list(self):
        return list(self.details_dict.values())

    def to_text(self, sep=","):
        return sep.join(self.details_dict.values())

    def to_file(self, directory_path, sep_text=",", file_name="details", format="txt"):
        file = open(os.path.join(directory_path, file_name + "." + format),"w")
        file.write(self.to_text(sep_text))
        file.close()

    @staticmethod
    def from_list(details_list):
        details = Details()
        if len(details_list) != len(details.details_dict.keys()):
            raise Exception("got " + str(len(details_list)) +
                            " but has to be " + str(len(details.details_dict.keys())))
        if details_list is not None:
            index = 0
            for title in details.details_dict.keys():
                details.details_dict[title] = details_list[index]
                index += 1
        return details

    @staticmethod
    def from_file(file_path, sep=","):
        file = open(file_path, "r")
        text = file.read()
        file.close()
        return Details.from_list(text.split(sep))


if __name__ == '__main__':
    details = Details.from_list(["a","b","c","d","e","f"])
    print(details["cityy"])