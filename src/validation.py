from pydantic import BaseModel

class FullInputFeatures(BaseModel):
    bedrooms: float
    bathrooms: float
    sqft_living: float
    sqft_lot: float
    floors: float
    waterfront: float
    view: float
    condition: float
    grade: float
    sqft_above: float
    sqft_basement: float
    yr_built: float
    yr_renovated: float
    zipcode: int
    lat: float
    long: float
    sqft_living15: float
    sqft_lot15: float

class InputFeatures(BaseModel):
    zipcode: int
    bedrooms: float
    bathrooms: float
    sqft_living: float
    sqft_lot: float
    floors: float
    sqft_above: float
    sqft_basement: float

