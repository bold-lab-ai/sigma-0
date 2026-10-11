
from enum import Enum

class MaxInt(Enum):
    _64_Bit_Signed=2_147_483_647
    _32_Bit_Signed=32_767

# Actual constants.
INITID=-2

# The "no order found" sentinel. It MUST stay outside the id space that
# init_msgs_from_l2 hands out, which is [INITID - (2*book_depth - 1), INITID].
#
# It was -99 (D-R6). That is outside the range at book_depth 10, and inside it
# from book_depth 49 upwards, so at the generator's book_depth of 500 a failed
# lookup named the real init order at L2 row 97 and cancel_order, which matches
# on id alone with no price constraint, removed quantity from it. 880 of 7200
# generated messages carried the sentinel, and the roughly half of them on the
# bid side each corrupted a level. It cost no replay fidelity, because both
# sides did it identically, which is exactly why nothing caught it.
#
# Far outside any plausible init range now, and JAXLOB_Configuration refuses to
# construct if a book_depth is ever chosen that would reach it, so this cannot
# come back quietly at some future depth.
NEGATIVE_RETURN_ID=-999_999_999

DUMMYID=-888888
EMPTY_SLOT=-1


ORDERBOOK_FEAT=6
TRADE_FEAT=8
NS_PER_SEC=1e9

# Default values for the config.

NTRADE_CAP=100
NORDER_CAP=100
STARTOFDAY=[34200 , 0]
ENDOFDAY=[57600 , 0]
TEST_TIME=[44444,44444]


# LOBSTER Message types, 
class MessageType(Enum):
    LIMIT=1  
    CANCEL=2 
    DELETE=3 
    MATCH=4
    HIDDEN=5
    AUCTION=6
    HALT=7

class OrderSideFeat(Enum):
    P=0 #Price
    Q=1 #Quantity
    OID=2 #Order ID
    TID=3 # Trade ID
    SEC=4 #Seconds
    NSEC=5 #Nanoseconds 

class TradesFeat(Enum):
    P=0 #Price
    Q=1 #Quantity
    PASS_OID=2 #Order ID
    AGRS_OID=3 # Trade ID
    SEC=4 #Seconds
    NSEC=5 #Nanoseconds 
    PASS_TID=6 #Order ID
    AGRS_TID=7 # Trade ID



class BidAskSide(Enum):
    BID=1
    ASK=-1


#Define as static...? Might solve 
class CancelMode(Enum):
    STRICT_BY_ID=0 #Cancel only if ID matches. 
    INCLUDE_INITS=1 #Cancel only init orders if ID does not match
    CANCEL_UNIFORM=2  #Pick a random order at the right price level to cancel 
    CANCEL_UNIFORM_AND_LARGE=3 # Unused for now

class Type4Interpretation(Enum):
    IOC=0 #Treat like a limit order that removes liquidity only. Unmatched quantiy IS NOT added to the book.
    LIM=1 #Treat like a limit order that removes and adds liquidity. Unmatched quantiy IS added to the book.
    MKT=2  #Treat like a limit order with extreme price (maxint for buys, 0 for sells) that removes liquidity only. Unmatched quantity IS NOT added to the book.
    #NOTE: MKT is catastrophic for speed: every order ends up being calculated as if it were a MO under vmap. 

SEED= 42 # the meaning of life. 

#TODO: flag on behaviour of type market either limit or far touch. 

class SimulatorMode(Enum):
    GENERAL_EXCHANGE=0
    LOBSTER_INTERPRETER=1

class LOBMSGFEAT(Enum):
    Type=0  # Message type
    Side=1  # Order side (buy/sell)
    Quant=2 # Quantity
    Price=3 # Price
    OID=4   # Order ID
    TID=5   # Trade ID
    TS=6    # Timestamp seconds
    TNS=7   # Timestamp nanoseconds