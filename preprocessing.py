from sklearn.base import BaseEstimator, TransformerMixin
from pydantic import BaseModel, Field
from typing import Optional

class CountryTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, country_counts):
        self.country_counts = country_counts

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.merge(self.country_counts[['country', 'country_grouped']], on='country', how='left')
        X = X.drop('country', axis=1)
        X['country_grouped'] = X['country_grouped'].fillna('Others')
        X.rename(columns={'country_grouped': 'country'}, inplace=True)
        return X
    

class BookingInput(BaseModel):
    hotel: str = Field(..., description="Hotel type, options: 'Resort Hotel', 'City Hotel'")
    lead_time: int = Field(..., description="Number of days between booking and arrival")
    arrival_date_month: str = Field(..., description="Month of arrival, options: 'January', 'February', etc.")
    arrival_date_week_number: int = Field(..., description="Week number of year for arrival date")
    arrival_date_day_of_month: int = Field(..., description="Day of month for arrival date")
    stays_in_weekend_nights: int = Field(..., description="Number of weekend nights (Saturday or Sunday) for the booking")
    stays_in_week_nights: int = Field(..., description="Number of week nights (Monday to Friday) for the booking")
    adults: int = Field(..., description="Number of adults")
    children: Optional[int] = Field(default=0, description="Number of children")
    babies: int = Field(..., description="Number of babies")
    meal: str = Field(..., description="Meal type, options: 'BB' (Bed & Breakfast), 'HB' (Half board), etc.")
    country: str = Field(..., description="Country of origin of the guest. Options: PRT, GBR, USA, ESP, etc.")
    market_segment: str = Field(..., description="Market segment, options: 'Online TA', 'Offline TA/TO', etc.")
    distribution_channel: str = Field(..., description="Distribution channel, options: 'TA/TO', 'Direct', etc.")
    is_repeated_guest: int = Field(..., description="0 if first time guest, 1 if repeated guest")
    previous_cancellations: int = Field(..., description="Number of previous bookings that were cancelled by the customer")
    previous_bookings_not_canceled: int = Field(..., description="Number of previous bookings that were not cancelled by the customer")
    reserved_room_type: str = Field(..., description="Code of room type reserved. Options: 'A', 'B', 'C', etc.")
    booking_changes: int = Field(..., description="Number of changes made to the booking")
    deposit_type: str = Field(..., description="Deposit type, options: 'No Deposit', 'Non Refund', 'Refundable'")
    days_in_waiting_list: int = Field(..., description="Number of days the booking was in the waiting list before confirmation")
    customer_type: str = Field(..., description="Customer type, options: 'Transient', 'Contract', etc.")
    adr: float = Field(..., description="Average Daily Rate (price per day of the booking)")
    required_car_parking_spaces: int = Field(..., description="Number of car parking spaces required")
    total_of_special_requests: int = Field(..., description="Number of special requests made by the customer")

    class Config:
        schema_extra = {
            "example": {
                "hotel": "Resort Hotel",
                "lead_time": 10,
                "arrival_date_month": "July",
                "arrival_date_week_number": 27,
                "arrival_date_day_of_month": 5,
                "stays_in_weekend_nights": 2,
                "stays_in_week_nights": 5,
                "adults": 2,
                "children": 1,
                "babies": 0,
                "meal": "BB",
                "country": "GBR",
                "market_segment": "Online TA",
                "distribution_channel": "TA/TO",
                "is_repeated_guest": 0,
                "previous_cancellations": 0,
                "previous_bookings_not_canceled": 0,
                "reserved_room_type": "A",
                "booking_changes": 1,
                "deposit_type": "No Deposit",
                "days_in_waiting_list": 0,
                "customer_type": "Transient",
                "adr": 150.0,
                "required_car_parking_spaces": 0,
                "total_of_special_requests": 2
            }
        }
