import pandas as pd

"""
    Represents one patient's prescription adherence profile.

    This class takes prescription records for a single patient and provides
    methods to summarize refill behavior, average medication cost, average
    days supplied, average proportion of days covered (PDC), and whether the
    patient meets a defined adherence threshold.
    """

class PatientAdherenceProfile:
    def __init__(self, patient_id, df: pd.DataFrame):
        self.patient_id = patient_id
        self.df = df.sort_values("fill_date")
        """
        Initialize a patient adherence profile.

        Parameters:
            patient_id: Unique identifier for the patient.
            df: DataFrame containing prescription records for this patient.
        """
    def total_fills(self):
        #TODO: Calculate and return the total number of fills?
        return len(self.df)
    """
        Count the total number of prescription fills for this patient.

        Returns:
            The number of prescription fill records.
        """
    def average_copay(self):
        #TODO: Calculate and return the average copay_amount
        return self.df['copay_amount'].mean()

    def average_days_supply(self):
        #TODO: Calculate and return the average days_supply
        return self.df['days_supply'].mean()

    def calculate_pdc(self):
        #TODO: Calculate and return the average proportion_days_covered (pdc)
        return self.df['proportion_days_covered'].mean()
    """
        Calculate the patient's average proportion of days covered (PDC).

        PDC is used as a medication adherence measure. Higher values indicate
        that the patient had medication available for a larger proportion of
        the observation period.

        Returns:
            Mean proportion_days_covered across all fills.
        """
    def is_adherent(self, threshold=0.75):
        #TODO: Return True if average pdc meets the adherence threshold provided
        return self.calculate_pdc() >= threshold
    
    def summary(self):
        return {
            "patient_id": self.patient_id,
            "total_fills": self.total_fills(),
            "avg_copay": self.average_copay(),
            "avg_days_supply": self.average_days_supply(),
            "pdc": self.calculate_pdc(),
            "adherent": self.is_adherent(),
        }
    """
        Create a dictionary summary of the patient's adherence profile.

        Returns:
            A dictionary containing patient ID, total fills, average copay,
            average days supply, average PDC, and adherence classification.
        """        