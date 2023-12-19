from enum import Enum


class Type(Enum):
    Regular = "Regular"
    Special = "Special"


kaif = Type.Regular

print(kaif.value)
