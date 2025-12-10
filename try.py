import numpy as np

class RectangleToSquareScaler:
    def __init__(self, a, b, c, d, f):
        """
        a, b : interval for x  (x ∈ [a, b])
        c, d : interval for y  (y ∈ [c, d])
        f    : original function f(x, y)
        """
        self.a, self.b = a, b
        self.c, self.d = c, d
        self.f = f

    # Map (u, v) from square to rectangle
    def x(self, u):
        return self.a + u * (self.b - self.a)

    def y(self, v):
        return self.c + v * (self.d - self.c)

    # Map (x, y) from rectangle back to square
    def u(self, x):
        return (x - self.a) / (self.b - self.a)

    def v(self, y):
        return (y - self.c) / (self.d - self.c)

    # Scaled function g(u, v) = f(x(u), y(v))
    def g(self, u, v):
        return self.f(self.x(u), self.y(v))


# ----------------------------------------------------
# EXAMPLE USAGE
# ----------------------------------------------------

# Define any original function f(x, y)
def f(x, y):
    return x*x * np.cos(y) +3

# Define the rectangle [a,b]×[c,d]
a, b = 2, 5
c, d = -1, 3

# Create the scaler
scaler = RectangleToSquareScaler(a, b, c, d, f)

# Evaluate the rescaled function on the unit square
u, v = 0.5, 0.25
print("g(u, v) =", scaler.g(u, v))

# Convert a point from rectangle to square
x, y = 4, 1
print("Mapped to square:", scaler.u(x), scaler.v(y))