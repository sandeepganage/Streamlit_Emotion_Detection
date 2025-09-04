import math
import streamlit as st

st.set_page_config(page_title="Number Inspector", page_icon="🔢", layout="centered")
st.title("🔢 Number Inspector — live on Streamlit Cloud")

st.write(
    "Type any integer and I’ll compute interesting properties in real time. "
    "This is perfect for a quick end-to-end ML/AI session kickoff on Streamlit Cloud."
)


# ---------------------------
# Utilities (pure Python)
# ---------------------------

def is_integer(s: str):
    try:
        int(s)
        return True
    except:
        return False


def is_prime(n: int) -> bool:
    """Deterministic trial division—fast enough for ~ up to 10^7 in a demo."""
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i, w = 5, 2
    # check i up to sqrt(n)
    while i * i <= n:
        if n % i == 0:
            return False
        i += w
        w = 6 - w
    return True


def prime_factors(n: int):
    """Return prime factorization as [(p, exp), ...]."""
    factors = []
    if n == 0 or n == 1 or n == -1:
        return factors
    x = abs(n)
    # factor out 2s
    cnt = 0
    while x % 2 == 0:
        x //= 2
        cnt += 1
    if cnt:
        factors.append((2, cnt))
    # factor odd
    f = 3
    while f * f <= x:
        cnt = 0
        while x % f == 0:
            x //= f
            cnt += 1
        if cnt:
            factors.append((f, cnt))
        f += 2
    if x > 1:
        factors.append((x, 1))
    return factors


def all_divisors(n: int):
    """All divisors of n (positive)."""
    if n == 0:
        return []  # infinite divisors; skip
    x = abs(n)
    small, large = [], []
    i = 1
    while i * i <= x:
        if x % i == 0:
            small.append(i)
            if i * i != x:
                large.append(x // i)
        i += 1
    return small + large[::-1]


def is_perfect_square(n: int) -> bool:
    if n < 0:
        return False
    r = int(math.isqrt(n))
    return r * r == n


def is_fibonacci(n: int) -> bool:
    # n is Fibonacci iff 5n^2 ± 4 is a perfect square
    return is_perfect_square(5 * n * n + 4) or is_perfect_square(5 * n * n - 4)


def is_armstrong(n: int) -> bool:
    # Narcissistic number in base 10
    x = abs(n)
    s = str(x)
    k = len(s)
    return sum(int(c) ** k for c in s) == x


def digital_sum(n: int) -> int:
    return sum(int(c) for c in str(abs(n)))


def digital_root(n: int) -> int:
    x = abs(n)
    return 0 if x == 0 else 1 + ((x - 1) % 9)


def sign_name(n: int) -> str:
    return "zero" if n == 0 else ("positive" if n > 0 else "negative")


def is_palindrome(n: int) -> bool:
    s = str(abs(n))
    return s == s[::-1]


def safe_factorial(n: int):
    """Limit factorial to n <= 1000 to avoid OOM."""
    if n < 0:
        return None, "Not defined for negative integers"
    if n > 1000:
        return None, "Too large (limit 1000) — use smaller n"
    # iterative factorial to avoid recursion depth
    res = 1
    for i in range(2, n + 1):
        res *= i
    return res, None


# ---------------------------
# Sidebar controls
# ---------------------------
st.sidebar.header("⚙️ Options")
compute_divisors = st.sidebar.checkbox("Compute all divisors (skip for very large n)", value=True)
compute_prime_factors = st.sidebar.checkbox("Compute prime factors", value=True)
show_factorial = st.sidebar.checkbox("Show factorial (limit 1000)", value=False)

# ---------------------------
# Input
# ---------------------------
default_str = "2025"
s = st.text_input("Enter an integer (e.g., 42, -17, 2025):", default_str)
if not is_integer(s):
    st.warning("Please enter a valid integer.")
    st.stop()

n = int(s)

# ---------------------------
# Basic properties
# ---------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Sign", sign_name(n))
col2.metric("Parity", "even" if n % 2 == 0 else "odd")
col3.metric("Prime?", "Yes ✅" if is_prime(n) else "No ❌")
col4.metric("Absolute value", f"{abs(n):,}")

# Numbers & checks
st.subheader("Computed values")
sq = n * n
cube = n * n * n
st.write(
    f"- Square: **{sq:,}**  \n"
    f"- Cube: **{cube:,}**  \n"
    f"- Palindrome? **{'Yes' if is_palindrome(n) else 'No'}**  \n"
    f"- Perfect square? **{'Yes' if is_perfect_square(n) else 'No'}**  \n"
    f"- Fibonacci member? **{'Yes' if is_fibonacci(n) else 'No'}**  \n"
    f"- Armstrong (narcissistic)? **{'Yes' if is_armstrong(n) else 'No'}**  \n"
    f"- Digital sum: **{digital_sum(n)}**, Digital root: **{digital_root(n)}**"
)

# Prime factors
if compute_prime_factors:
    st.subheader("Prime factorization")
    if n in (0, 1, -1):
        st.info(f"{n} has no standard prime factorization.")
    else:
        pf = prime_factors(n)
        if not pf:
            st.write("No factors found (this shouldn’t happen unless n is ±1 or 0).")
        else:
            pretty = " × ".join(f"{p}^{e}" if e > 1 else f"{p}" for p, e in pf)
            st.write(f"**{n}** = **{pretty}**")

# All divisors
if compute_divisors:
    st.subheader("All divisors (positive)")
    if n == 0:
        st.info("0 has infinitely many divisors; skipping.")
    else:
        divs = all_divisors(n)
        st.write(f"Count: **{len(divs)}**")
        st.write(divs[:300] if len(divs) > 300 else divs)

# Factorial (optional)
if show_factorial:
    st.subheader("Factorial")
    fact, err = safe_factorial(n)
    if err:
        st.error(err)
    else:
        st.code(f"{n}! = {fact}")

st.caption(
    "Tip: Try values like 97 (prime), 121 (perfect square & palindrome), "
    "153 (Armstrong), 34 (Fibonacci), 2025 (fun one!)."
)
