from pydantic import BaseModel


class CustomerFeatures(BaseModel):
    Amount: float
    Value: float
    FraudResult: int
    ProviderId_ProviderId_2: bool = False
    ProviderId_ProviderId_3: bool = False
    ProviderId_ProviderId_4: bool = False
    ProviderId_ProviderId_5: bool = False
    ProviderId_ProviderId_6: bool = False
    ProductCategory_data_bundles: bool = False
    ProductCategory_financial_services: bool = False
    ProductCategory_movies: bool = False
    ProductCategory_other: bool = False
    ProductCategory_ticket: bool = False
    ProductCategory_transport: bool = False
    ProductCategory_tv: bool = False
    ProductCategory_utility_bill: bool = False
    ChannelId_ChannelId_2: bool = False
    ChannelId_ChannelId_3: bool = False
    ChannelId_ChannelId_5: bool = False
    PricingStrategy_1: bool = False
    PricingStrategy_2: bool = False
    PricingStrategy_4: bool = False