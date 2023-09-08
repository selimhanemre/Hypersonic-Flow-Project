import numpy as np
import math
import matplotlib.pyplot as plt

# Given variables from Student IDs
t = 0.1  # thickness ratio
M_inf = 7  # Freestream Mach number
gamma = 1.1  # Specific Heat Ratio

R = (1 + t**2)/(4*t)  # Radius of the Circle that defines the Circular Arch Airfoil

# Initial Points Adjustments
N = 24  # number of initialized points
r = 0.025  # distance from origin (leading edge)
iterations = 400  # number of max loops in method of charecteristics
edge_clearance = 0.01  # edge clearence for the shock and wall

# Free Flow Characteristics (assume sea level ISA standards)
P_inf = 101.325  # pressure (kPa)
T_inf = 288  # temperature (K)
rho_inf = 1.225  # density (kg/m3)


def circular_arch(x):

    # Circular Arch Airfoil function
    y = np.sqrt(R**2 - (x-0.5)**2) - (R - (t/2))

    return y


def circular_arch_slope(x):

    # Derivative of the Circular Arch Airfoil Function
    dy_dx = (0.5-x)/(np.sqrt((R**2) - ((x-0.5)**2)))

    return dy_dx


def prandtl_meyer(M, gamma):
    # Prandtl-Meyer function
    return np.sqrt((gamma + 1) / (gamma - 1)) * np.arctan(np.sqrt((gamma - 1) / (gamma + 1) * (M**2 - 1))) - np.arctan(np.sqrt(M**2 - 1))


def find_mach_number(target_prandtl_meyer, gamma, a=0.1, b=30, tolerance=1e-6, max_iterations=100):
    # Find the input value (Mach number) that produces the desired target Prandtl Meyer
    # Using the bisection method
    if prandtl_meyer(a, gamma) == target_prandtl_meyer:
        return a
    if prandtl_meyer(b, gamma) == target_prandtl_meyer:
        return b

    for _ in range(max_iterations):
        # Calculate the midpoint and Prandtl Meyer at midpoint
        c = (a + b) / 2
        PM = prandtl_meyer(c, gamma)

        # Check if c is the solution
        if np.isclose(PM, target_prandtl_meyer, atol=tolerance):
            return c

        # Update the interval based on the sign of the function value at c
        if PM < target_prandtl_meyer:
            a = c
        else:
            b = c

    # If no solution is found within the maximum iterations
    raise RuntimeError(
        "Mach not found within the maximum number of iterations.")


def theta_from_beta(beta):

    # Theta Beta Mach Relation
    theta = math.atan(
        ((2/math.tan(beta))*(((M_inf**2)*math.sin(beta)**2-1)) / ((M_inf**2)*(gamma+math.cos(2*beta))+2)))

    return theta


def beta_from_theta(theta, a=1 * np.pi/180, b=70 * np.pi/180, tolerance=1e-9, max_iterations=100):

    # Reversed Theta Beta Mach Relation using Bisection
    if theta_from_beta(a) == theta:
        return a
    if theta_from_beta(b) == theta:
        return b

    for _ in range(max_iterations):
        # Calculate the midpoint
        c = (a + b) / 2

        # Check if solution is close enough
        if np.abs(theta_from_beta(c) - theta) < tolerance:
            return c

        # Update the interval based on the value at c
        if theta_from_beta(c) < theta:
            a = c
        else:
            b = c

    # If no solution is found within the iterations
    return 0


def find_linear_intersection(x1, y1, m1, x2, y2, m2):
    # Intersection of two lines in the form of: y - y0 = m0(x - x0)

    # Solve the equations for x-coordinate
    x3 = (y2 - y1 - m2*x2 + m1*x1) / (m1 - m2)

    # Calculate the y-coordinate of the intersection point using either line equation
    y3 = m1*(x3 - x1) + y1

    return x3, y3


def find_arch_intersection(x0, y0, m0, tolerance=1e-6, max_iterations=100):
    # Find the intersection point of the neg char line and circular arch using the bisection method

    # The negative characteristic line
    def line(x):
        return m0*(x - x0) + y0

    a = 0.0  # Initial lower bound for x
    b = 1.0  # Initial upper bound for x

    for _ in range(max_iterations):
        c = (a + b) / 2  # midpoint of bounds

        # Evaluate the functions at the midpoint and adjust the interval
        if line(c) - circular_arch(c) < 0:
            b = c
        else:
            a = c

        # Check for convergence
        if abs(line(c) - circular_arch(c)) < tolerance:
            return c, line(c)

    # If no solution is found within the iterations
    # This indicates thet the calculation point has passed the trailing edge
    return 0, 0


def interior_point(x1, y1, M1, Theta1, x2, y2, M2, Theta2):

    # Find Mach Anges
    Mangle1 = np.arcsin(1 / M1)
    Mangle2 = np.arcsin(1 / M2)

    # Find Prandtl Meyer angles
    PM1 = prandtl_meyer(M1, gamma)
    PM2 = prandtl_meyer(M2, gamma)

    # Find k konstant for characteristic lines from compatibility equations
    K1neg = Theta1 + PM1
    K2pos = Theta2 - PM2

    # Use these constants to find PM and Theta for point 3
    Theta3 = (K1neg + K2pos)/2
    PM3 = (K1neg - K2pos)/2

    # Use Prandtl-Meyer3 to find Mach Angle 3
    M3 = find_mach_number(PM3, gamma)
    Mangle3 = np.arcsin(1/M3)

    # Find slope of lines connecting the points
    m1 = np.arctan(((Theta1 - Mangle1) + (Theta3 - Mangle3)) / 2)
    m2 = np.arctan(((Theta2 +
                     Mangle2) + (Theta3 + Mangle3)) / 2)

    # Find the intersection: x3 and y3\
    x3, y3 = find_linear_intersection(x1, y1, m1, x2, y2, m2)

    return x3, y3, M3, Theta3


def wall_point(x4, y4, M4, Theta4, max_iterations=100):

    # Calculate Mach Angle and Prandtl-Meyer angle at point 4
    Mangle4 = np.arcsin(1/M4)
    PM4 = prandtl_meyer(M4, gamma)

    # Initial guess for point 5 values
    M5 = M4
    Theta5 = Theta4
    Mangle5 = np.arcsin(1 / M5)

    # Loop to find the location of point 5
    for _ in range(max_iterations):
        m4 = np.tan(((Theta4 - Mangle4) + (Theta5 - Mangle5)) / 2)

        x5, y5 = find_arch_intersection(x4, y4, m4)
        if x5 == 0:
            return 0, 0, 0, 0

        Theta5 = np.arctan(circular_arch_slope(x5))

        PM5 = Theta4 + PM4 - Theta5
        M5 = find_mach_number(PM5, gamma)
        if (abs(Mangle5 - np.arcsin(1 / M5)) < 1e-9):
            return x5, y5, M5, Theta5
        Mangle5 = np.arcsin(1 / M5)

    raise RuntimeError(
        "No solution found within the maximum number of iterations .")


def mach_difference_given_beta(Theta6, PM6, Beta7):
    # The Difference between the Mach number calculated with the two methods

    # Calculate M7 using the charecteristic Equations
    Theta7 = theta_from_beta(Beta7)
    PM7 = Theta7 - Theta6 + PM6
    M7_char = find_mach_number(PM7, gamma)

    # Calculate M7 using shock wave relations
    Mn_inf = M_inf*math.sin(Beta7)
    Mn_7 = np.sqrt(((gamma-1)*(Mn_inf**2) + 2) /
                   ((2*gamma*Mn_inf**2) - (gamma-1)))
    M7_shock = Mn_7 / math.sin(Beta7 - Theta7)

    # The Difference:
    return M7_char - M7_shock


def shock_point(x6, y6, M6, Theta6, x8, y8, Beta8, max_iterations=100, tolerance=1e-6):

    # Calculate Prantdl Meyer angle and Mach angle
    PM6 = prandtl_meyer(M6, gamma)
    Mangle6 = np.arcsin(1 / M6)

    # Beta 7 will be calculated with Bisection Method
    # Beta 7 will be around Beta8
    a = Beta8 + (1 * math.pi/180)  # upper bound for Beta 7
    b = Beta8 - (4 * math.pi/180)  # lower bound for Beta 7

    # Check if Bisection method is viable
    if np.sign(mach_difference_given_beta(Theta6, PM6, a)) == np.sign(mach_difference_given_beta(Theta6, PM6, b)):
        raise ValueError(
            "Error values at a and b must have opposite signs.")

    # Check the validity of the boundaries first
    if mach_difference_given_beta(Theta6, PM6, a) == 0:
        Beta7 = a
    if mach_difference_given_beta(Theta6, PM6, b) == 0:
        Beta7 = b
    else:
        # Bisection Loop:
        for _ in range(max_iterations):
            c = (a + b) / 2
            Beta7 = c

            if np.isclose(mach_difference_given_beta(Theta6, PM6, c), 0, atol=tolerance):
                break

            if np.sign(mach_difference_given_beta(Theta6, PM6, c)) == np.sign(mach_difference_given_beta(Theta6, PM6, a)):
                a = c
            else:
                b = c

    # Calculate Theta for determined Beta angle and freeflow Mach
    Theta7 = theta_from_beta(Beta7)

    # Calculate Mach 7 using charecteristic equations
    PM7 = Theta7 - Theta6 + PM6
    M7 = find_mach_number(PM7, gamma)

    # Mach angle at point 7
    Mangle7 = np.arcsin(1/M7)

    # angle of linearized char line connecting point 6 and 7
    m6 = ((Theta6 + Mangle6) + (Theta7 + Mangle7))/2

    # Intersection of Shock Wave and char line
    x7, y7 = find_linear_intersection(x6, y6, m6, x8, y8, np.tan(Beta8))

    return x7, y7, M7, Theta7, Beta7


def Method_of_Characteristics():
    # The first two sets of points will be initialized before the main loop
    x = np.zeros([2, N])
    y = np.zeros([2, N])
    M = np.zeros([2, N])
    Theta = np.zeros([2, N])

    # The shock angles will be kept track of in a seperate array
    Beta = np.zeros(1)

    # Define Theta and Beta at the leading edge
    Theta0 = np.arctan(circular_arch_slope(0))
    Beta0 = beta_from_theta(Theta0)

    # Calculate Mach using oblique shock wave relations
    Mn_inf = M_inf*math.sin(Beta0)
    Mn_0 = math.sqrt(((gamma-1)*(Mn_inf**2) + 2) /
                     ((2*gamma*Mn_inf**2) - (gamma-1)))
    M0 = Mn_0 / math.sin(Beta0 - Theta0)

    # # # Calculate the minimum and maximum angle for the initial points # # #
    # Theta min is at the point on the airfoil with r distance from origin

    def difference_between_arch_y(x):
        y_r = np.sqrt(r**2 - x**2)
        y_airfoil = circular_arch(x)
        return y_r - y_airfoil

    # Calculate X min using bisection method
    a = 0
    b = r
    for _ in range(50):
        # Calculate the midpoint
        X_min = (a + b) / 2

        # Check if c is the solution
        if np.abs(difference_between_arch_y(X_min)) < 1e-9:
            break

        # Update the interval based on the sign of the function value at c
        if difference_between_arch_y(X_min) > 0:
            a = X_min
        else:
            b = X_min

    # Define the min and max angles for initial points
    Theta_min = np.arccos(X_min/r)
    Theta_max = Beta0

    # Initialize the first set of points all interior points
    for i in range(N-1):

        # evenly distributed points in a circular arch with defined edge clearance
        x[0][i] = r * np.cos(Theta0 + (((Theta_max - Theta_min) *
                                        (i + edge_clearance)) / ((N-1) + (2 * edge_clearance))))
        y[0][i] = r * np.sin(Theta0 + (((Theta_max - Theta_min) *
                                        (i + edge_clearance)) / ((N-1) + (2 * edge_clearance))))

        # The initial points assumed to have the Mach and Theta of the leading edge
        M[0][i] = M0
        Theta[0][i] = Theta0

    # Second set of points will include one wall and one shock point
    for i in range(N):
        if i == 0:  # Lower-most point is a wall point
            x[1][i], y[1][i], M[1][i], Theta[1][i] = wall_point(
                x[0][i], y[0][i], M[0][i], Theta[0][i])
        elif i == N-1:  # Upper-most point is a shock point
            x[1][i], y[1][i], M[1][i], Theta[1][i], Beta[0] = shock_point(
                x[0][i-1], y[0][i-1], M[0][i-1], Theta[0][i-1], 0, 0, Beta0)
        else:  # Others are all interior points
            x[1][i], y[1][i], M[1][i], Theta[1][i] = interior_point(
                x[0][i], y[0][i], M[0][i], Theta[0][i], x[0][i-1], y[0][i-1], M[0][i-1], Theta[0][i-1])

    # Has the latest wall point surpassed the trailing edge?
    Passed_Trailing_Edge = False

    # j represents the set of points used to calculate the next set
    for j in range(1, iterations):
        x_next = np.zeros(N)
        y_next = np.zeros(N)
        M_next = np.zeros(N)
        Theta_next = np.zeros(N)

        Beta_next = 0

        # Point sets with an even j have one wall and one shock point
        if j % 2 == 0:
            for i in range(N):
                if i == 0:
                    x_next[i], y_next[i], M_next[i], Theta_next[i] = wall_point(
                        x[j][i], y[j][i], M[j][i], Theta[j][i])
                    if x_next[i] == 0:
                        # Check if the edge is surpassed
                        Passed_Trailing_Edge = True
                        break
                elif i == N-1:
                    x_next[i], y_next[i], M_next[i], Theta_next[i], Beta_next = shock_point(
                        x[j][i-1], y[j][i-1], M[j][i-1], Theta[j][i-1], x[j-1][i], y[j-1][i], Beta[(int(j/2)-1)])
                    Beta = np.append(Beta, Beta_next)
                else:
                    x_next[i], y_next[i], M_next[i], Theta_next[i] = interior_point(
                        x[j][i], y[j][i], M[j][i], Theta[j][i], x[j][i-1], y[j][i-1], M[j][i-1], Theta[j][i-1])
            if Passed_Trailing_Edge:
                # Break out of the loop if the airfoil has been completely calculated
                break
        # Point sets with an odd j have only interior points
        else:
            for i in range(N - 1):
                x_next[i], y_next[i], M_next[i], Theta_next[i] = interior_point(
                    x[j][i+1], y[j][i+1], M[j][i+1], Theta[j][i+1], x[j][i], y[j][i], M[j][i], Theta[j][i])

        # Add the next set of points into the matrices
        x = np.vstack((x, x_next))
        y = np.vstack((y, y_next))
        M = np.vstack((M, M_next))
        Theta = np.vstack((Theta, Theta_next))

    if Passed_Trailing_Edge:
        print("Loop DID reach the trailing edge")
    else:
        print("Loop DID NOT reach the trailing edge")

    return x, y, M, Theta


def calculate_pres_temp_dens(x, y, M, Theta):
    # Assume flow is isentropic

    # Freestream total pres and temp (isentropic relations)
    Pt_inf = P_inf * np.power((1 + ((gamma - 1) / 2)
                              * M_inf**2), (gamma / (gamma - 1)))
    Tt_inf = T_inf * (1 + ((gamma - 1) / 2) * M_inf**2)

    # Calculate Mach Normal to Leading Edge Shock Wave
    Theta0 = np.arctan(circular_arch_slope(0))
    Beta0 = beta_from_theta(Theta0)
    Mn_inf = M_inf*math.sin(Beta0)

    # Total pres and temp after shockwave
    Pt_0 = Pt_inf
    Pt_0 *= np.power((((gamma + 1) * Mn_inf**2) /
                     (((gamma - 1) * Mn_inf**2) + 2)), (gamma / (gamma - 1)))
    Pt_0 *= np.power(((gamma + 1) / ((2 * gamma * Mn_inf**2) -
                     (gamma - 1))), (1 / (gamma - 1)))

    Tt_0 = Tt_inf

    # Use Isentropic equations to calculate pres and temp at all points
    P = Pt_0 / np.power((1 + ((gamma - 1) / 2) *
                        np.power(M, 2)), (gamma / (gamma - 1)))
    T = Tt_0 / (1 + ((gamma - 1) / 2) * np.power(M, 2))

    # Ideal Gas equation to calculate density
    R = 8.314  # Ideal gas constant in J/(mol·K)

    rho = (P / (R * T))

    # Delete empty points
    P[np.where(M == 0)] = 0
    T[np.where(M == 0)] = 0
    rho[np.where(M == 0)] = 0

    return P, T, rho


def surface_distribution(A):
    # extract wall points of matrix A

    # initialize array
    a = np.zeros(0)

    for i in range(len(A)-1):
        if (A[i][N-1] == 0):
            a = np.append(a, A[i+1, 0])
    return a


def newtonian_methods(t, M, gamma):

    R = (1 + t**2)/(4*t)

    # Defining Theta in terms of Chord (0 -> 1)
    x = np.linspace(0, 1, 41)  # x = [0, 0.025, 0.05 ... 0.95, 0.975, 1]
    Theta = np.zeros(0)
    for i in range(len(x)):
        Theta = np.append(Theta, -math.asin((x[i]-0.5)/R))

    # Newtonian CP Calculation
    CP_newtonian = np.zeros(0)
    for i in range(len(Theta)):
        CP_newtonian = np.append(
            CP_newtonian, 2*math.pow(math.sin(Theta[i]), 2))

    # Find CP max for modification
    CP_max = (2/(gamma*M**2))*((math.pow((((gamma+1)**2)*(M**2)) /
                                         (4*gamma*M**2-2*(gamma-1)), gamma/(gamma-1))*((1-gamma+2*gamma*M**2)/(gamma+1)))-1)

    # Modified Newtonian CP Calculation
    CP_modified = np.zeros(0)
    for i in range(len(Theta)):
        CP_modified = np.append(CP_modified, CP_max *
                                math.pow(math.sin(Theta[i]), 2))

    # Determine y coordinate of points using Theta
    gamma = np.zeros(0)
    for i in range(len(Theta)):
        gamma = np.append(
            gamma, (R*math.cos(Theta[i])) - (R-(t/2)))

    # Determine the derivative of Theta
    dTheta_dy = np.zeros(0)
    for i in range(len(Theta)):
        if i < len(Theta)/2:
            dTheta_dy = np.append(
                dTheta_dy, (Theta[i+1] - Theta[i])/(gamma[i+1] - gamma[i]))
        else:
            dTheta_dy = np.append(
                dTheta_dy, (Theta[i] - Theta[i-1])/(gamma[i] - gamma[i-1]))

    # Centrifugal Corrected CP Calculation
    CP_centrifugal = np.zeros(0)
    CP_mod_centrifugal = np.zeros(0)
    for i in range(len(Theta)):
        I = 0  # the integration term must be calculated with a loop
        for j in range(i):
            I += math.cos(Theta[i]) * (gamma[j+1]-gamma[j])
        correction = 2*dTheta_dy[i]*math.sin(Theta[i])*I
        CP_centrifugal = np.append(
            CP_centrifugal, CP_newtonian[i] + correction)
        CP_mod_centrifugal = np.append(
            CP_mod_centrifugal, CP_modified[i] + correction)

    return x[0:21], CP_newtonian[0:21], CP_modified[0:21], CP_mod_centrifugal[0:21]


def CP_Method_of_Characteristics(gamma, P_inf, P, M):
    CP_MoC = (P-P_inf) / ((gamma * P_inf * M**2) / 2)
    return CP_MoC


def plot_contours(x, y, M, title, label):
    # Plot contour in the area below the shockwave

    # Calculate coordinates for the Airfoil
    x_circular_arch = np.linspace(0, 1, 100)
    y_circular_arch = circular_arch(x_circular_arch)

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot The Airfoil
    ax.plot(x_circular_arch, y_circular_arch, color='blue', label='Airfoil')

    # turn matricies into arrays for easy plotting
    x = x.flatten()
    y = y.flatten()
    M = M.flatten()

    # Remove the zero points
    x = x[np.nonzero(x)]
    y = y[np.nonzero(y)]
    M = M[np.nonzero(M)]

    # Calculate min and max values
    vmin = np.min(M)
    vmax = np.max(M)

    # Define the colormap
    cmap = plt.cm.get_cmap('coolwarm')

    # Plot the points with color-coded temperatures
    plt.scatter(x, y, c=M, cmap=cmap, vmin=vmin, vmax=vmax)

    # Set colorbar
    cbar = plt.colorbar()
    cbar.set_label(label)

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)

    # Add legend
    ax.legend(loc='upper left')

    # Show the plot
    plt.ylim(0, 0.12)
    plt.xlim(0, 1)
    plt.show()


def plot_surface_distribution(x, distribution, title, label):

    # Create a scatter plot with lines connecting the points
    plt.plot(x, distribution, 'k-')
    plt.scatter(x, distribution, marker='None')

    # Set the axis labels
    plt.xlabel('x')
    plt.ylabel(label)
    plt.title(title)

    plt.xlim([0, 1])

    # Show the plot
    plt.show()


def plot_two_surface_distribution(x1, distribution1, x2, distribution2, title, axis_label, label1, label2):
    # Create a scatter plot with lines connecting the points
    plt.plot(x1, distribution1, label=label1)
    plt.plot(x2, distribution2, label=label2)
    plt.plot([0, 1], [0, 0])

    # Set the axis labels
    plt.xlabel('x')
    plt.ylabel(axis_label)
    plt.title(title)

    plt.xlim([0, 1])

    # Add legend
    plt.legend(loc="upper right")

    # Show the plot
    plt.show()


# Calculate points in the control volume and their properties
x, y, M, Theta = Method_of_Characteristics()
P, T, rho = calculate_pres_temp_dens(x, y, M, Theta)

# Plot the Contours of the Properties in the Control Volume
plot_contours(x, y, M, 'Mach Number Contours', 'Mach Number')
plot_contours(x, y, P, 'Pressure Contours', 'Pressure (kPa)')
plot_contours(x, y, T, 'Temperature of Contours', 'Temperature (K)')
plot_contours(x, y, rho, 'Density of Contours', 'Density (kg/m3)')

# Extract the fluid proporties on the surface of the airfoil
x_surface = surface_distribution(x)
M_surface = surface_distribution(M)
P_surface = surface_distribution(P)
T_surface = surface_distribution(T)
rho_surface = surface_distribution(rho)

# Plot these Proporties as Surface Distributions
plot_surface_distribution(x_surface, P_surface,
                          'Surface Pressure Distribution', 'Pressure (kPa)')
plot_surface_distribution(
    x_surface, T_surface, 'Surface Temperature Distribution', 'Temperature (K)')
plot_surface_distribution(x_surface, rho_surface,
                          'Surface Density Distribution', 'Density (kg/m3)')

# Calculate the Coefficient of Pressure with values from MoC
CP_MoC = CP_Method_of_Characteristics(gamma, P_inf, P_surface, M_inf)

# Calculate CP with the three newtonian methods
X_newtonian, CP_newtonian, CP_modified, CP_mod_centrifugal = newtonian_methods(
    t, M_inf, gamma)

# Plot these methods in comparison
plot_two_surface_distribution(X_newtonian, CP_newtonian, x_surface, CP_MoC,
                              "MoC vs Newtonian CP", "CP", 'Newtonian', 'Method of Charecteristics')
plot_two_surface_distribution(X_newtonian, CP_modified, x_surface, CP_MoC,
                              "MoC vs Modified Newtonian CP", "CP", 'Modified Newtonian', 'Method of Charecteristics')
plot_two_surface_distribution(X_newtonian[0:15], CP_mod_centrifugal[0:15], x_surface, CP_MoC, "MoC vs Modified Newtonian with Centrifugal Correction CP",
                              "CP", 'Modified Newtonian with Centrifugal Correction', 'Method of Charecteristics')
