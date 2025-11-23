class modify_control:
    """
    change control inputs in the dataframe by a given percentage
    controls to be changed are aps, pbrake_f, pbrake_r
    """

    def __init__(self, original_df, percent_change=0.1):
        self.df = original_df.copy()
        self.percent = percent_change
        self.controls = ['aps', 'pbrake_f', 'pbrake_r']

    def get_range(self, control_input):
        if control_input not in self.controls:
            raise ValueError(f"{control_input} is not a valid control")
        col = self.df[control_input]
        return col.min(), col.max()

    def modify_one(self, control_input):
        if control_input not in self.controls:
            raise ValueError(f"{control_input} is not a valid control")

        df_mod = self.df.copy()
        df_mod[control_input] = df_mod[control_input] * (1 + self.percent)
        return df_mod

    def modify_all(self):
        modified = {}
        for c in self.controls:
            modified[c] = self.modify_one(c)
        return modified
