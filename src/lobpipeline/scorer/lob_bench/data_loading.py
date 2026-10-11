import os
import pandas as pd
import numpy as np
import decimal
from decimal import Decimal
from typing import Optional
import glob
import os
from tqdm import tqdm
from dataclasses import dataclass
import statsmodels.api as sm
import warnings
import re

# --- Polars auto-detection for fast CSV parsing ---
_USE_POLARS = False
try:
    import polars as pl
    _USE_POLARS = True
except ImportError:
    pass


def get_price_range_for_level(
        book: pd.DataFrame,
        lvl: int
    ) -> pd.DataFrame:
    assert lvl > 0
    assert lvl <= (book.shape[1] // 4)
    p_range = book[[(lvl-1) * 4, (lvl-1) * 4 + 2]].copy()
    p_range.columns = ['p_max', 'p_min']
    # replace -1 nan value with same nan values used in lobster data
    p_range.p_max = p_range.p_max.replace(-1, 9999999999)
    p_range.p_min = p_range.p_min.replace(-1, -9999999999)
    return p_range

def filter_by_lvl(
        messages: pd.DataFrame,
        book: pd.DataFrame,
        lvl: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:

    assert messages.shape[0] == book.shape[0]
    book = book.iloc[:, :lvl * 4]
    p_range = get_price_range_for_level(book, lvl)
    messages = messages[(messages.price <= p_range.p_max) & (messages.price >= p_range.p_min)]
    book = book.loc[messages.index]
    return messages, book

def cut_data_to_lvl(in_path, out_path, lvl, overwrite=False):
    if not overwrite and (in_path == out_path):
        raise ValueError('in_path and out_path are the same, set overwrite=True to overwrite files')

    book_paths = sorted(glob.glob(in_path + '*orderbook*.csv'))
    message_paths = sorted(glob.glob(in_path + '*message*.csv'))
    assert len(book_paths) == len(message_paths)

    for m_f, b_f in zip(message_paths, book_paths):
        b = load_book_df(b_f)
        m = load_message_df(m_f, parse_time=False)
        # print('before', m.shape, b.shape)

        m, b = filter_by_lvl(m, b, lvl)
        # print('after', m.shape, b.shape)
        # print('save to', out_path + m_f.rsplit('/', 1)[1])
        # print()
        m.to_csv(out_path + m_f.rsplit('/', 1)[1], header=None, index=None)
        b.to_csv(out_path + b_f.rsplit('/', 1)[1], header=None, index=None)

def _load_message_df_polars(m_f: str, parse_time: bool = True) -> pd.DataFrame:
    """Fast CSV parsing via Polars, with pandas boundary for Decimal time."""
    cols = ['time', 'event_type', 'order_id', 'size', 'price', 'direction']
    schema = {
        'time': pl.Utf8,
        'event_type': pl.Int32,
        'order_id': pl.Int32,
        'size': pl.Int32,
        'price': pl.Int32,
        'direction': pl.Int32,
    }
    try:
        df_pl = pl.read_csv(
            m_f,
            has_header=False,
            new_columns=cols,
            schema_overrides=schema,
            ignore_errors=True,
            truncate_ragged_lines=True,
            n_columns=6,
        )
    except Exception:
        return _load_message_df_pandas(m_f, parse_time)

    int_cols = ['event_type', 'order_id', 'size', 'price', 'direction']
    df_pl = df_pl.drop_nulls(subset=int_cols)
    # Build pandas from column arrays (avoids pyarrow dependency)
    messages = pd.DataFrame({c: df_pl[c].to_numpy() for c in df_pl.columns})

    if parse_time:
        try:
            messages.time = messages.time.apply(
                lambda x: Decimal(
                    re.sub(
                        r"(\.)(?=.*\.)",
                        '',
                        re.sub('[^0-9,.]', '', str(x))
                    )
                )
            )
        except decimal.InvalidOperation as e:
            print("error with file ", m_f)
            print("can't convert times to decimal")
            print("times:")
            print(messages.time)
            raise e
    return messages


def _load_book_df_polars(b_f: str) -> pd.DataFrame:
    """Fast CSV parsing via Polars for orderbook files."""
    try:
        df_pl = pl.read_csv(b_f, has_header=False, ignore_errors=True)
        return pd.DataFrame({c: df_pl[c].to_numpy() for c in df_pl.columns})
    except Exception:
        return _load_book_df_pandas(b_f)


def _load_message_df_pandas(m_f: str, parse_time: bool = True) -> pd.DataFrame:
    """Original pandas CSV parsing path."""
    cols = ['time', 'event_type', 'order_id', 'size', 'price', 'direction']
    messages = pd.read_csv(
        m_f,
        names=cols,
        usecols=cols,
        index_col=False,
        on_bad_lines='skip',
        dtype={
            'time': str,
            # 'event_type': 'int32',
            # 'order_id': 'int32',
            # 'size': 'int32',
            # 'price': 'int32',
            # 'direction': 'int32'
        }
    )
    # convert columns data types
    for col in ['event_type', 'order_id', 'size', 'price', 'direction']:
        messages[col] = pd.to_numeric(messages[col], errors='coerce', downcast='integer')
        # drop nan values produced by coercion
        messages = messages.dropna(how='any', subset=col)
        # try again to parse to int
        messages[col] = messages[col].astype('int32')

    # messages["event_type"] = pd.to_numeric(messages["event_type"], errors='coerce', downcast='integer')
    # messages["order_id"] = pd.to_numeric(messages["order_id"], errors='coerce', downcast='integer')
    # messages["size"] = pd.to_numeric(messages["size"], errors='coerce', downcast='integer')
    # messages["price"] = pd.to_numeric(messages["price"], errors='coerce', downcast='integer')
    # messages["direction"] = pd.to_numeric(messages["direction"], errors='coerce', downcast='integer')
    # messages = messages.dropna()

    if parse_time:
        try:
            messages.time = messages.time.apply(
                lambda x: Decimal(
                    re.sub(
                        r"(\.)(?=.*\.)", # remove multiple decimal points
                        '',
                        re.sub('[^0-9,.]', '', x)  # remove all non-numeric characters except commas and periods
                    )
                )
            )
        except decimal.InvalidOperation as e:
            print("error with file ", m_f)
            print("can't convert times to decimal")
            print("times:")
            print(messages.time)
            raise e
    return messages


def _load_book_df_pandas(b_f: str) -> pd.DataFrame:
    """Original pandas CSV parsing path."""
    return pd.read_csv(b_f, index_col=False, header=None)


def _load_message_df_npy(m_f: str, parse_time: bool = True) -> pd.DataFrame:
    """Load message from .npy — 10x faster than CSV (no parse/regex overhead).
    Preserves Decimal precision for time field (same as CSV path).

    NPY column layout (14 cols):
        0=ORDER_ID, 1=EVENT_TYPE, 2=DIRECTION, 3=PRICE_ABS, 4=unused,
        5=SIZE, 6=unused, 7=unused, 8=TIMEs, 9=TIMEns, 10-13=unused
    """
    arr = np.load(m_f)
    time_s = arr[:, 8].astype(np.int64)
    time_ns = arr[:, 9].astype(np.int64)
    direction = arr[:, 2].astype(np.int32)
    direction[direction == 0] = -1
    # Build time as Decimal directly from int components (faster than CSV regex path)
    if parse_time:
        times = [Decimal(f'{s}.{str(ns).zfill(9)}') for s, ns in zip(time_s, time_ns)]
    else:
        times = time_s.astype(np.float64) + time_ns * 1e-9
    messages = pd.DataFrame({
        'time': times,
        'event_type': arr[:, 1].astype(np.int32),
        'order_id': arr[:, 0].astype(np.int32),
        'size': arr[:, 5].astype(np.int32),
        'price': arr[:, 3].astype(np.int32),
        'direction': direction,
    })
    return messages


def _load_book_df_npy(b_f: str) -> pd.DataFrame:
    """Load orderbook from .npy — direct numpy array to DataFrame."""
    return pd.DataFrame(np.load(b_f))


def load_message_df(m_f: str, parse_time: bool = True) -> pd.DataFrame:
    # Direct .npy path
    if m_f.endswith('.npy'):
        return _load_message_df_npy(m_f, parse_time)
    # Prefer .npy sidecar if available (same name, .csv → .npy)
    npy_f = m_f.rsplit('.csv', 1)[0] + '.npy'
    if os.path.exists(npy_f):
        return _load_message_df_npy(npy_f, parse_time)
    if _USE_POLARS:
        return _load_message_df_polars(m_f, parse_time)
    return _load_message_df_pandas(m_f, parse_time)

def load_book_df(b_f: str) -> pd.DataFrame:
    # Direct .npy path
    if b_f.endswith('.npy'):
        return _load_book_df_npy(b_f)
    # Prefer .npy sidecar if available
    npy_f = b_f.rsplit('.csv', 1)[0] + '.npy'
    if os.path.exists(npy_f):
        return _load_book_df_npy(npy_f)
    # Polars is slower than pandas for wide CSVs (500 cols orderbook),
    # so always use pandas for book loading.
    return _load_book_df_pandas(b_f)

def add_date_to_time(df: pd.DataFrame, date: str) -> pd.Series:
    # df.time = pd.to_datetime(date) + pd.to_datetime(df.time, unit='s')
    sec_nanosec = df.time.astype(str).str.split('.', expand=True)
    sec_nanosec[1] = sec_nanosec[1].str.pad(9, side='right', fillchar='0')
    df.time = pd.to_datetime(date) + pd.to_timedelta(sec_nanosec[0] + 's') + pd.to_timedelta(sec_nanosec[1] + 'ns')
    return df


class Lazy_Tuple():
    """ Takes callables as args, returning their results when indexed. """
    def __init__(self, *args) -> None:
        self.args = args

    def __getitem__(self, i):
        return self.args[i]()

    def __len__(self):
        return len(self.args)


class Lobster_Sequence():
    def __init__(
            self,
            date: str,
            real_id: int,
            m_real: callable,
            b_real: callable,
            num_gen_series: tuple[int],
            m_gen: Optional[tuple[callable]] = None,
            b_gen: Optional[tuple[callable]] = None,
            m_cond: Optional[tuple[callable]] = None,
            b_cond: Optional[tuple[callable]] = None,
            # num_gen_series,
            # m_gen = None,
            # b_gen = None,
            # m_cond = None,
            # b_cond = None,
            # NOTE uncomment this return back when publish,
            # tuple[int] cannot be recognized on my env
            # it was commented for easy tesging.
        ) -> None:

        self.date = date
        self.real_id=real_id

        if callable(m_real):
            self._m_real = m_real
        else:
            self.m_real = m_real

        if callable(b_real):
            self._b_real = b_real
        else:
            self.b_real = b_real

        if m_gen is not None:
            if callable(m_gen[0]):
                self._m_gen = m_gen
            else:
                self.m_gen = m_gen
        else:
            self.m_gen = None

        if b_gen is not None:
            if callable(b_gen[0]):
                self._b_gen = b_gen
            else:
                assert len(b_gen) == num_gen_series[0], f"Expected {num_gen_series[0]} generated book files. Got {len(b_gen)}."
                self.b_gen = b_gen
        else:
            self.b_gen = None

        if callable(m_cond):
            self._m_cond = m_cond
        else:
            self.m_cond = m_cond

        if callable(b_cond):
            self._b_cond = b_cond
        else:
            self.b_cond = b_cond

        self.num_gen_series = num_gen_series

    def materialize(self):
        """ Replace callables with their results. Can be useful if data loading functions take a long time. """
        self.m_real = self.m_real
        self.b_real = self.b_real
        self.m_cond = self.m_cond
        self.b_cond = self.b_cond
        if self._m_gen is not None:
            self.m_gen = tuple(m for m in self.m_gen)
        if self._b_gen is not None:
            self.b_gen = tuple(b for b in self.b_gen)

    @property
    def m_real(self):
        x = self._m_real()
        # if not isinstance(x, tuple):
        #     x = (x,)
        return x

    @m_real.setter
    def m_real(self, value):
        if value is None:
            self._m_real = None
        if callable(value):
            self._m_real = value
        else:
            self._m_real = lambda: value

    @property
    def b_real(self):
        x = self._b_real()
        # if not isinstance(x, tuple):
        #     x = (x,)
        return x

    @b_real.setter
    def b_real(self, value):
        if value is None:
            self._b_real = None
        elif callable(value):
            self._b_real = value
        else:
            self._b_real = lambda: value

    @property
    def m_gen(self):
        if self._m_gen is None:
            return None
        return Lazy_Tuple(*self._m_gen)

    @m_gen.setter
    def m_gen(self, value):
        if value is None:
            self._m_gen = None
        elif callable(value):
            self._m_gen = value
        else:
            self._m_gen = tuple(lambda: x for x in value)

    @property
    def b_gen(self):
        if self._b_gen is None:
            return None
        return Lazy_Tuple(*self._b_gen)

    @b_gen.setter
    def b_gen(self, value):
        if value is None:
            self._b_gen = None
        elif callable(value):
            self._b_gen = value
        else:
            self._b_gen = tuple(lambda: x for x in value)

    @property
    def m_cond(self):
        return self._m_cond()

    @m_cond.setter
    def m_cond(self, value):
        if value is None:
            self._m_cond = None
        elif callable(value):
            self._m_cond = value
        else:
            self._m_cond = lambda: value

    @property
    def b_cond(self):
        return self._b_cond()

    @b_cond.setter
    def b_cond(self, value):
        if value is None:
            self._b_cond = None
        elif callable(value):
            self._b_cond = value
        else:
            self._b_cond = lambda: value


class Simple_Loader():
    def __init__(
            self,
            real_data_path: str,
            gen_data_path: str,
            cond_data_path: str
        ) -> None:
        # Try npy files first (10x faster than CSV), fall back to CSV
        real_message_paths = sorted(glob.glob(real_data_path + '/*_message.npy'))
        if len(real_message_paths) > 0:
            self.is_npy = True
            real_book_paths = sorted(glob.glob(real_data_path + '/*_orderbook.npy'))
        else:
            self.is_npy = False
            real_message_paths = sorted(glob.glob(real_data_path + '/*message*.csv'))
            real_book_paths = sorted(glob.glob(real_data_path + '/*orderbook*.csv'))
        assert len(real_message_paths) > 0, f"No real message files found in {real_data_path}"
        if len(real_book_paths) == 0:
            real_book_paths = [None] * len(real_message_paths)

        self.paths = []
        for rmp, rbp, in tqdm(zip(real_message_paths, real_book_paths)):
            fname = rmp.rsplit('/', maxsplit=1)[-1]
            date_str = fname.split('_')[1]

            if self.is_npy:
                # NPY naming: GOOG_2026-01-02_real_id_548_message.npy
                #              GOOG_2026-01-02_real_id_548_gen_id_0_message.npy
                m_real_id = re.search(r'real_id_(\d+)', fname)
                assert m_real_id, f"Cannot parse real_id from npy filename: {fname}"
                real_id = m_real_id.group(1)

                gen_messsage_paths = sorted(glob.glob(gen_data_path + f'/*{date_str}*real_id_{real_id}_gen_id_*_message.npy'))
                if len(gen_messsage_paths) == 0:
                    # Fall back to CSV gen files if no npy gen files
                    gen_messsage_paths = sorted(glob.glob(gen_data_path + f'/*{date_str}*message*real_id_{real_id}_gen_id_*.csv'))
                    if len(gen_messsage_paths) == 0:
                        warnings.warn(f"No generated message files found for real_id={real_id}")

                gen_book_paths = sorted(glob.glob(gen_data_path + f'/*{date_str}*real_id_{real_id}_gen_id_*_orderbook.npy'))
                if len(gen_book_paths) == 0:
                    gen_book_paths = sorted(glob.glob(gen_data_path + f'/*{date_str}*orderbook*real_id_{real_id}_gen_id_*.csv'))

                cond_message_path = sorted(glob.glob(cond_data_path + f'/*{date_str}*real_id_{real_id}_message.npy'))
                if len(cond_message_path) == 0:
                    cond_message_path = sorted(glob.glob(cond_data_path + f'/*{date_str}*message*real_id_{real_id}.csv'))
                cond_book_path = sorted(glob.glob(cond_data_path + f'/*{date_str}*real_id_{real_id}_orderbook.npy'))
                if len(cond_book_path) == 0:
                    cond_book_path = sorted(glob.glob(cond_data_path + f'/*{date_str}*orderbook*real_id_{real_id}.csv'))
            else:
                # CSV naming: GOOG_2026-01-02_message_real_id_1032.csv
                before, _, after = rmp.partition('real_id_')
                real_id = after.split('_')[0].split('.')[0]

                gen_messsage_paths = sorted(glob.glob(gen_data_path + f'/*{date_str}*message*real_id_{real_id}_gen_id_*.csv'))
                if len(gen_messsage_paths) == 0:
                    warnings.warn(f"No generated message files found for real_id={real_id}")
                gen_book_paths = sorted(glob.glob(gen_data_path + f'/*{date_str}*orderbook*real_id_{real_id}_gen_id_*.csv'))

                cond_message_path = sorted(glob.glob(cond_data_path + f'/*{date_str}*message*real_id_{real_id}.csv'))
                cond_book_path = sorted(glob.glob(cond_data_path + f'/*{date_str}*orderbook*real_id_{real_id}.csv'))

            if len(cond_message_path) == 0:
                cond_message_path = None
            elif len(cond_message_path) == 1:
                cond_message_path = cond_message_path[0]
            else:
                print(cond_message_path)
                raise ValueError(f"Multiple conditional message files found. (real_id={real_id})")

            if len(cond_book_path) == 0:
                cond_book_path = None
            elif len(cond_book_path) == 1:
                cond_book_path = cond_book_path[0]
            else:
                print(cond_book_path)

                raise ValueError(f"Multiple conditional book files found. (real_id={real_id})")

            self.paths.append((date_str, real_id,rmp, rbp, gen_messsage_paths, gen_book_paths, cond_message_path, cond_book_path))

    def __len__(self) -> int:
        return len(self.paths)

    # def __getitem__(self, i: int):
    def __getitem__(self, i: int) -> tuple[pd.DataFrame, pd.DataFrame, tuple[pd.DataFrame], Optional[tuple[pd.DataFrame]]]:
        """ Get 1 real and N generated dataframes for a given period
            Returns: real_messages, real_book, tuple(gen_messages), [tuple(gen_books)]
            The generated book files are optional and will be calculated from messages using JaxLob simulator if not provided.
        """

        date, real_id ,rmp, rbp, gmp, gbp, cmp, cbp = self.paths[i]

        def m_real():
            m = load_message_df(rmp)
            m = add_date_to_time(m, date)
            return m

        def b_real():
            return load_book_df(rbp)

        m_gen = tuple(
            lambda: add_date_to_time(
                load_message_df(m),
                date
            ) for m in gmp
        )

        b_gen = tuple(lambda: load_book_df(b) for b in gbp)

        def m_cond():
            if cmp is not None:
                m = load_message_df(cmp)
                if not m.empty:
                    m = add_date_to_time(m, date)
                return m
            else:
                return None

        def b_cond():
            if cbp is not None:
                return load_book_df(cbp)
            else:
                return None

        s = Lobster_Sequence(
            date,
            real_id,
            m_real,
            b_real,
            num_gen_series=(len(gmp),),
            m_gen=m_gen,
            b_gen=b_gen,
            m_cond=m_cond,
            b_cond=b_cond,
        )

        return s


if __name__=='__main__':
    d = os.environ.get('LOB_BENCH_DATA', './data_lob_bench/')
    loader = Simple_Loader(d+'data_test_real', d+'data_test_gen', d+'data_test_cond')
    loader[0]
    print()