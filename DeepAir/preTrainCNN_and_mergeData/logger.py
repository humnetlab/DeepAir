testGroup = 9


class Logger:
    def __init__(self, save_dirs, data_years, exp_name, testGroup):
        self.file = open(
            "./{}/{}/{}_G{}.log".format(
                save_dirs, data_years, exp_name, str(testGroup)
            ),
            "w",
        )

    def log(self, content):
        self.file.write(content + "\n")
        self.file.flush()
