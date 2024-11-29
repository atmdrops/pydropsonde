class QualityControl:
    """
    Helper class to handle quality control functions and flags in a sonde object
    """

    def __init__(
        self,
    ) -> None:
        self.qc_vars = []
        self.qc_flags = {}
        self.qc_details = {}
        self.qc_by_var = {}

    def set_qc_variables(self, qc_variables):
        self.qc_vars = self.qc_vars + list(qc_variables)
        for variable in self.qc_vars:
            self.qc_by_var.update({variable: dict(qc_flags={}, qc_details={})})
