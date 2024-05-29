from datetime import date, datetime
from tracemalloc import start
from typing import Any, List, Tuple, Union
from dateutil.relativedelta import relativedelta
import numpy as np
from numpy import ndarray, add, subtract, multiply, divide, power, concatenate, full, arange, sum
import pandas as pd

class Bond():
    """models a bond"""

    def __init__(self,
                 issuance_date: str, # yyyy-mm-dd
                 coupon_rate: float, # per annum
                 maturity_date: Union[str, None] = None, #yyyy-mm-dd
                 payment_dates: Union[List[date], None] = None, # [yyyy-mm-dd]
                 principal: float = 100.,
                 frequency: int = 1, # number of interest payments per annum,
                 structure: str = "bullet", # cash flow structure; can be one of: bullet, amortization, zero-coupon, etc.
            ) -> None:
        """sets the features of the bond"""

        if issuance_date is None:
            raise ValueError("Issuance date must be provided.")
        
        if maturity_date is None and payment_dates is None:
            raise ValueError("""
                             Either issuance_date and maturity_date, or payment_dates must be provided.

                             If settlement_date and maturity_date are given, then coupon and principal payments are being inferred by the frequency of interest payments.
                             For example, if settlement_date is 2024-01-01 and frequency is 2 (semi-annual interest payments), then the first coupon payment
                             is going to be 6 months later, at 2024-07-01, the second coupon payment 6 months after the first coupon payment, at 2025-01-01, and so on.
                             The last coupon payment and the repayment of principal is assumed that is going to happen at the maturity_date.

                             If payment_days are given, it must conform to the following format:
                             [first_coupon_payment_date, second_coupon_payment_date, ..., last_coupon_and_principal_payment_date],
                             where each date should have the format: 'yyyy-mm-dd'.

                             If both issuance_date, maturity_date and payment_dates are provided, then only payment_dates are used.
                             """)

        self.issuance_date: date = datetime.fromisoformat(issuance_date).date()
        self.principal: float = principal
        self.coupon_rate: float = coupon_rate # interest rate per annum
        self.frequency: float = frequency
        self.structure: str = structure
        self.coupon: float = multiply(divide(self.coupon_rate, self.frequency), self.principal)
        self.maturity_date: date = payment_dates[-1] if maturity_date is None and payment_dates is not None else datetime.fromisoformat(maturity_date).date()
        self.payment_dates: List[date] = self._determine_payment_dates() if payment_dates is None else payment_dates
        self.tenor: int = self.maturity_date.year - self.issuance_date.year
        self.cash_flows: ndarray = self._calculate_cash_flows()

    def _determine_payment_dates(self,) -> List[date]:
        """determines the interest and principal payments of the bond"""
        payment_dates: List[date] = []
        current_date: date = self.issuance_date
        while current_date < self.maturity_date:
            current_date += relativedelta(months = int(12 / self.frequency))
            payment_dates.append(current_date)
        return payment_dates

    def _calculate_cash_flows(self,) -> ndarray:
        """determines the cash flows of the bond"""
        # using more convinient notation
        n: int = len(self.payment_dates) # number of payments (coupon and principal)
        c: float = self.coupon
        p: float = self.principal
        if self.structure == "bullet":
            return concatenate((full(subtract(n, 1), c), add(c, p)), axis = None)
        return np.array([])

    def _calculate_discount_factors(self,
                                    index_of_next_payment: int,
                                    discount_rate: float # discount rate per annum
                                ) -> ndarray:
        """calculates the discount factors based on the given discount rate"""
        r: float = discount_rate / self.frequency # discount rate per payment period; that is, if payment frequency is semi-annual, then r = discount_rate / frequency, where frequency = 2
        n: int = len(self.cash_flows[index_of_next_payment:])
        return divide(1, power(add(1, r), arange(1, n + 1)))

    def _count_days(self,
                    starting_date: date, # start of period
                    ending_date: date, # end of period
                    basis: str, # actual/actual or 30/360
                ) -> int:
        """counts the number of days between the starting and the ending date, using the basis method"""
        if basis == "30/360":
            return (
                ((ending_date.day if ending_date.day != 31 else 30) - (starting_date.day if starting_date.day != 31 else 30)) + \
                ((ending_date.month - starting_date.month) * 30) + \
                ((ending_date.year - starting_date.year) * 360)
            )
        else:
            return (ending_date - starting_date).days

    def _find_previous_and_next_payment_dates(self,
                                              settlement_date: date,
                                            ) -> Tuple[int, Tuple[date, date]]:
        index_of_last_payment: int = len(self.cash_flows) - 1
        for index, payment_date in enumerate(self.payment_dates):
            if settlement_date == payment_date:
                if index > 0:
                    return index + 1, (self.payment_dates[index], self.payment_dates[index + 1])
                else:
                    return index + 1, (self.payment_dates[index], self.payment_dates[index + 1])
            elif settlement_date < payment_date:
                if 0 < index < index_of_last_payment:
                    return index, (self.payment_dates[index - 1], self.payment_dates[index])
                else:
                    if index:
                        return index_of_last_payment, (self.payment_dates[index_of_last_payment - 1], self.payment_dates[index_of_last_payment])
                    else:
                        return index, (self.issuance_date, self.payment_dates[index])
        return -1, (date.today(), date.today())

    def _calculate_accrued_interest(self,
                                    settlement_date: date,
                                    previous_payment_date: date,
                                    next_payment_date: date,
                                    basis: str,
                                ) -> Tuple[float, float]:
        """calculates the accrual factor and interest for a given transaction date"""
        try:
            accrual_factor: float = self._count_days(previous_payment_date, settlement_date, basis) / self._count_days(previous_payment_date, next_payment_date, basis)
            accrued_interest: float = self.coupon * accrual_factor
        except ZeroDivisionError:
            accrual_factor: float = 0.
            accrued_interest: float = 0.
        return accrued_interest, accrual_factor

    def _calculate_present_value(self,
                                 cash_flows: ndarray,
                                 discount_factors: ndarray
                                ) -> float:
        """calculates present value of cash flows, given discount factors"""
        return sum(multiply(discount_factors, cash_flows), axis = 0)
    
    def _calculate_future_value(self,
                                cash_flows: ndarray,
                                premium_factors: ndarray
                            ) -> float:
        """calculates future value of cash flows, given premium factors"""
        return sum(multiply(premium_factors, cash_flows), axis = 0)

    def calculate_price(self,
                        settlement_date: date,
                        basis: str, # actual/actual, 30/360
                        discount_rate: float,
                    ) -> Union[Tuple[float, float], ValueError]:
        """calculates the price of the bond"""
        if settlement_date < self.issuance_date or settlement_date > self.maturity_date:
            return ValueError("settlement date can't be before issuance date or after maturity date")
        elif settlement_date == self.maturity_date:
            return self.principal, self.principal
        else:
            index_of_next_payment, (previous_payment_date, next_payment_date) = self._find_previous_and_next_payment_dates(settlement_date)
            accrued_interest, accrual_factor = self._calculate_accrued_interest(settlement_date, previous_payment_date, next_payment_date, basis)
            cash_flows: ndarray = self.cash_flows[index_of_next_payment:]
            discount_factors: ndarray = self._calculate_discount_factors(index_of_next_payment, discount_rate)
            present_value: float = self._calculate_present_value(cash_flows, discount_factors)
            full_price: float = self._calculate_future_value(
                premium_factors = power(add(1, divide(discount_rate, self.frequency)), accrual_factor),
                cash_flows = np.array(present_value)
            )
            flat_price: float = subtract(full_price, accrued_interest)
            return full_price, flat_price

    def summarize_structure_of_cash_flows(self,) -> pd.DataFrame:
        return pd.DataFrame({"Date": self.payment_dates, "Cash Flow": self.cash_flows,})
