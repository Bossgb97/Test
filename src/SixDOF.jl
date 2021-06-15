module SixDOF

using LinearAlgebra: norm, cross
using Plots

export MassProp, State
export Atmosphere
#export sixdof!
# ------------ GENERIC MODULES -------------------------------------------------
import Dierckx
import CSV
import DataFrames
import JLD
import Dates
using PyPlot
using LinearAlgebra: cross, I

# ------------ FLOW CODES ------------------------------------------------------
# FLOWVLM https://github.com/byuflowlab/FLOWVLM
import FLOWVLM
const vlm = FLOWVLM

# FLOWVPM https://github.com/byuflowlab/FLOWVPM.jl
try                     # Load FLOWVPM if available
    import FLOWVPM
catch e                 # Otherwise load a dummy version of FLOWVPM
    @warn("FLOWVPM module not found. Using dummy module instead.")
    include("FLOWUnsteady_dummy_FLOWVPM.jl")
end
const vpm = FLOWVPM

# GeometricTools https://github.com/byuflowlab/GeometricTools.jl
import GeometricTools
const gt = GeometricTools

import FLOWNoise
const noise = FLOWNoise

# BPM https://github.com/byuflowlab/BPM.jl
import BPM

# ------------ GLOBAL VARIABLES ------------------------------------------------
const module_path = splitdir(@__FILE__)[1]                # Path to this module
const def_data_path = joinpath(module_path, "../data/")   # Default path to data folder


# ------------ HEADERS ---------------------------------------------------------
# Load modules
#for module_name in ["vehicle", "vehicle_vlm",
#                    "maneuver", "rotor",
#                    "simulation_types", "simulation", "utils",
#                    "processing", "monitors",
#                    "noise_wopwop", "noise_bpm"]
                    include("FLOWUnsteady_vehicle.jl")
                    include("FLOWUnsteady_vehicle_vlm.jl")
                    include("FLOWUnsteady_maneuver.jl")
                    include("FLOWUnsteady_rotor.jl")
                    include("FLOWUnsteady_simulation_types.jl")
                    include("FLOWUnsteady_simulation.jl")
                    include("FLOWUnsteady_utils.jl")
                    include("FLOWUnsteady_processing.jl")
                    include("FLOWUnsteady_monitors.jl")
                    include("FLOWUnsteady_noise_wopwop.jl")
                    include("FLOWUnsteady_noise_bpm.jl")
#end

#end # END OF MODULE


# ------ General Structs -------

"""
    State(x, y, z, phi, theta, psi, u, v, w, p, q, r)

State of the aircraft: positions in inertial frame, euler angles,
velocities in body frame, angular velocities in body frame.
"""
struct State{TF}
    x::TF  # position (inertial frame)
    y::TF
    z::TF
    phi::TF  # orientation, euler angles
    theta::TF
    psi::TF
    u::TF  # velocity (body frame)
    v::TF
    w::TF
    p::TF  # angular velocity (body frame)
    q::TF
    r::TF
end

"""
    Control(de, dr, da, df, throttle)

Define the control settings: delta elevator, delta rudder, delta aileron,
delta flaps, and throttle.
"""
#struct Control{TF}
#    de::TF  # elevator
#    dr::TF  # rudder
#    da::TF  # aileron
#    df::TF  # rudder
#    throttle::TF
#end

"""
    MassProp(m, Ixx, Iyy, Izz, Ixz, Ixy, Iyz)

Mass and moments of inertia in the body frame.
Ixx = int(y^2 + z^2, dm)
Ixz = int(xz, dm)

We can model our quadcopter as two thin uniform rods crossed at the origin with a point mass
(motor) at the end of each. With this in mind, it’s clear that the symmetries result in a diagonal
inertia matrix
"""
struct MassProp{TF}
    m::TF
    Ixx::TF
    Iyy::TF
    Izz::TF
end



"""
    Reference(S, b, c)

The reference area equal for each rotor, S is the reference area (propeller cross-section, not
area swept out by the propeller)
"""
#struct Reference{TF}
    #S::TF  # area
    #b::TF  # span
    #c::TF  # chord
#end

# ----------------------------------------------


# --------------- Interfaces ---------------

#abstract type AbstractAtmosphereModel end

"""
inizialization of the state position
"""
#function pos(state)
#     x = [state.x; state.y; state.z]
#    Wi = [0.0, 0.0, 0.0]
#    Wb = [0.0, 0.0, 0.0]
#    return x
#end

"""
    inizialization of the state velocities
"""
#function vel(state)
#     xdot = [state.u; state.v; state.w]
#    rho = 1.225  # sea-level properties
#    asound = 340.3
#    return xdot
#end

"""
    inizialization of the state angles
"""
#function angle(state)
#    theta = [state.phi; state.theta; state.psi]
#    g = 9.81
#    return theta
#end


# ----

#abstract type AbstractAeroModel end

"""

"""
#function anglevel(state)
#    omega = [state.p; state.q; state.r]
    # forces and moments in body frame
#    F = [0.0, 0.0, 0.0]
#    M = [0.0, 0.0, 0.0]
#    return omega
#end

# ----

#abstract type AbstractPropulsionModel end

"""
    Every variable used here is a matrix with the three axis for rows and time
    for columns
"""
function inizialization(N)
    row = 3
    matrix = zeros((row,N))
    return matrix
end
#    @warn "propulsionforces function not implemented for AbstractPropulsionModel"
    # forces and moments in body frame
#    F = [0.0, 0.0, 0.0]
#    M = [0.0, 0.0, 0.0]
#    return F, M
#end

# ----

#abstract type AbstractInertialModel end

"""
    gravityforces(model::AbstractInertialModel, atm::AbstractAtmosphereModel, state::State, control::Control, mp::MassProp, ref::Reference)

Compute the gravitational forces and moments in the body frame.
return F, M
"""
function equations(dt, i, mp, atm, k, b, kd, rotors, c, l, xprev, omegaprev, thetaprev, xdotprev)
    #the cycle for is initialized outside and then the function equations is called
        a = Array{Float64}(undef, 3)
        a = acceleration(i, thetaprev, xdotprev, mp, atm, kd, rotors, k)
        omegadot = Array{Float64}(undef, 3)
        #omegadot = omegadot(i, o, mp, l, k, b)
        I = [mp.Ixx   0.0   0.0;
             0.0     mp.Iyy 0.0;
             0.0      0.0  mp.Izz]
        tau = torque(i, l, b, k)
        #Inv = [1/I[1,1];1/I[2,2];1/I[3,3]]
        #prod = [tau[1]*Inv[1];tau[2]*Inv[2];tau[3]*Inv[3]]
        #diff = [((I[2,2]-I[3,3])/I[1,1])*o[2]*o[3];((I[3,3]-I[1,1])/I[2,2])*o[1]*o[3];
        #((I[1,1]-I[2,2])/I[3,3])*o[1]*o[2]]
        #omegadot = prod - diff
        omegadot = inv(I) * (tau - cross(omegaprev, I * omegaprev))
        omega = Array{Float64}(undef, 3)
        omega = omegaprev + dt * omegadot #i am not sure if they are
        #written well for the matrices
        thetadot = Array{Float64}(undef, 3)
        thetadot = omega2thetadot(thetaprev, omega)
        theta = Array{Float64}(undef, 3)
        theta = thetaprev + dt * thetadot
        xdot = Array{Float64}(undef, 3)
        xdot = xdotprev + dt * a
        x = Array{Float64}(undef, 3)
        x = xprev + dt * xdot
        return a ,omegadot, omega, thetadot, theta, xdot, x, tau
end

#    @warn "gravityforces function not implemented for AbstractInertialModel"
    # forces and moments in body frame
#    F = [0.0, 0.0, 0.0]
#    M = [0.0, 0.0, 0.0]
#    return F, M
#end


# ----

#abstract type AbstractController end

"""
    setcontrol(controller::AbstractController, time, atm::AbstractAtmosphereModel, state::State, lastcontrol::Control, mp::MassProp, ref::Reference)

Compute control state for next time step given current state
return control::Control
"""
#function setcontrol(controller::AbstractController, time, atm, state, mp, ref)
#    @warn "setcontrol function not implemented for AbstractController"
#    control = Control(0.0, 0.0, 0.0, 0.0, 0.0)
#    return control
#end


# -----------------------------




# ------------- helper functions (private) --------------


"""
    inertialtobody(state)

Construct a rotation matrix from inertial frame to body frame

The assumed order of rotation is
1) psi radians about the z axis,
2) theta radians about the y axis,
3) phi radians about the x axis.

This is an orthogonal transformation so its inverse is its transpose.
"""
function inertialtobody(theta)

    R = Array{eltype(theta)}(undef,3, 3)

    cphi, ctht, cpsi = cos.([theta[1], theta[2], theta[3]])
    sphi, stht, spsi = sin.([theta[1], theta[2], theta[3]])

    R[1, 1] = ctht*cpsi
    R[1, 2] = ctht*spsi
    R[1, 3] = -stht

    R[2, 1] = sphi*stht*cpsi - cphi*spsi
    R[2, 2] = sphi*stht*spsi + cphi*cpsi
    R[2, 3] = sphi*ctht

    R[3, 1] = cphi*stht*cpsi + sphi*spsi
    R[3, 2] = cphi*stht*spsi - sphi*cpsi
    R[3, 3] = cphi*ctht

    return R

end

function bodytoinertial(theta)

    Ri = Array{eltype(theta)}(undef,3, 3)

    cphi, ctht, cpsi = cos.([theta[1], theta[2], theta[3]])
    sphi, stht, spsi = sin.([theta[1], theta[2], theta[3]])

    Ri[1, 1] = cphi*cpsi - ctht*sphi*spsi
    Ri[1, 2] = -cpsi*sphi - cphi*ctht*spsi
    Ri[1, 3] = stht*spsi

    Ri[2, 1] = ctht*sphi*cpsi + cphi*spsi
    Ri[2, 2] = cphi*ctht*cpsi - sphi*spsi
    Ri[2, 3] = -cpsi*stht

    Ri[3, 1] = sphi*stht
    Ri[3, 2] = cphi*stht
    Ri[3, 3] = ctht

    return Ri

end



"""
time of simulation giving the the initial and final time and the time step
"""
function time(tinit, tfinal, dt);
    n = (tfinal-tinit)/dt
    n = trunc(Int,n)
    t = tinit
    v = Array{Float64}(undef, n)
    v[1] = t
    for i in 2: n
        v[i] = v[i-1] + dt
    end
return v,n

end



"""
 Total thrust, giving the square rotation velocity and the coefficient k
"""
function thrust(rpm, k, rotors)
sum=0
for i in 1:rotors
    sum  = sum + rpm[i]
end
    T = [0.0; 0.0; k*sum]
    return T
end


"""
Torque, giving the square rotation velocity, the distance of rotors from the center
and the coefficients k and b
"""
function torque(rpm, l, b, k)
    tau = [l*k*(rpm[1]-rpm[3]); l*k*(rpm[2]-rpm[4]); b*(rpm[1]-rpm[2]+rpm[3]-rpm[4])]
    return tau
end


# ----------------------------------------------------


# ------- Some Default Interface Implementations -----

"""
    StabilityDeriv(CL0, CLalpha, CLq, CLM, CLdf, CLde, alphas,
        CD0, U0, exp_Re, e, Mcc, CDdf, CDde, CDda, CDdr,
        CYbeta, CYp, CYr, CYda, CYdr, Clbeta,
        Clp, Clr, Clda, Cldr,
        Cm0, Cmalpha, Cmq, CmM, Cmdf, Cmde,
        Cnbeta, Cnp, Cnr, Cnda, Cndr)

Stability derivatives of the aircraft.  Most are self explanatory if you are
familiar with stability derivatives (e.g., CLalpha is dCL/dalpha or the lift curve slope).
Some less familiar ones include
- M: Mach number
- alphas: the angle of attack for stall
- U0: the speed for the reference Reynolds number CD0 was computed at
- exp_Re: the coefficient in the denominator of the skin friction coefficient (0.5 laminar, 0.2 turbulent)
- e: Oswald efficiency factor
- Mcc: crest critical Mach number (when compressibility drag rise starts)

"""
#struct StabilityDeriv{TF} <: AbstractAeroModel
#    CL0::TF
#    CLalpha::TF
#    CLq::TF
#    CLM::TF
#    CLdf::TF
#    CLde::TF
#    alphas::TF  # TODO: should probably do in terms of CLmax

#    CD0::TF
#    U0::TF  # velocity corresponding to Reynolds number of CD0  (TODO: rethink this)
#    exp_Re::TF  # exponent for Reynolds number scaling. typical values: exp_Re = 0.5 laminar, 0.2 turbulent
#    e::TF  # Oswald efficiency factor
#    Mcc::TF  # crest-critical Mach number when compressibility drag rise starts (quartic)
#    CDdf::TF
#    CDde::TF
#    CDda::TF
#    CDdr::TF

#    CYbeta::TF
#    CYp::TF
#    CYr::TF
#    CYda::TF
#    CYdr::TF

#    Clbeta::TF
#    Clp::TF
#    Clr::TF
#    Clda::TF
#    Cldr::TF

#    Cm0::TF
#    Cmalpha::TF
#    Cmq::TF
#    CmM::TF
#    Cmdf::TF
#    Cmde::TF

#    Cnbeta::TF
#    Cnp::TF
#    Cnr::TF
#    Cnda::TF
#    Cndr::TF
#end


"""
Computation of the linear acceleration
"""
function acceleration(rpm, theta, xdot, mp, atm, kd, rotors, k)
    gravity= [0.0; 0.0; -atm.g]
    R = bodytoinertial(theta)
    Tb = thrust(rpm, k, rotors)
    Ti = R * Tb
    Fd = -kd * xdot
    a = gravity + 1/mp.m  * Ti + 1/mp.m  * Fd
return a
end
    # Mach number and dynamic pressure
#    rho, asound = properties(atm, state)
#    Mach = Va / asound
#    qdyn = 0.5 * rho * Va^2

    # rename for convenience
#    p = state.p
#    q = state.q
#    r = state.r
#    de = control.de
#    df = control.df
#    dr = control.dr
#    da = control.da


    # lift
#    CL = sd.CL0 + sd.CLalpha*alpha + sd.CLq*q *ref.c/(2*Va) + sd.CLM*Mach
#        + sd.CLdf*df + sd.CLde*de

#    em = exp(-50*(alpha - sd.alphas))
#    ep = exp(50*(alpha + sd.alphas))
#    sigma = (1 + em + ep)/((1 + em)*(1 + ep))
#    CL = (1- sigma)*CL + sigma * 2 * sign(alpha)*sin(alpha)^2*cos(alpha)

    # drag
#    CDp = sd.CD0*(Va/sd.U0)^sd.exp_Re
#    CDi = CL^2/(pi*(ref.b^2/ref.S)*sd.e)
#    CDc = (Mach < sd.Mcc) ? 0.0 : 20*(Mach - sd.Mcc)^4

#    CD = CDp + CDi + CDc + abs(sd.CDdf*df) + abs(sd.CDde*de) + abs(sd.CDda*da) + abs(sd.CDdr*dr)

    # side force
#    CY = sd.CYbeta*beta + (sd.CYp*p + sd.CYr*r)*ref.b/(2*Va) + sd.CYda*da + sd.CYdr*dr

    # rolling moment
#    Cl = sd.Clbeta*beta + (sd.Clp*p + sd.Clr*r)*ref.b/(2*Va) + sd.Clda*da + sd.Cldr*dr

    # pitching moment
#    Cm = sd.Cm0 + sd.Cmalpha*alpha + sd.Cmq*q * ref.c/(2*Va) + sd.CmM*Mach + sd.Cmdf*df + sd.Cmde*de

    # yawing moment
#    Cn = sd.Cnbeta*beta + (sd.Cnp*p + sd.Cnr*r)*ref.b/(2*Va) + sd.Cnda*da + sd.Cndr*dr

    # transfer forces from wind to body axes
#    Rwb = windtobody(alpha, beta)

#    F = Rwb*[-CD, CY, -CL] * qdyn * ref.S

#    M = Rwb*[Cl*ref.b, Cm*ref.c, Cn*ref.b] * qdyn * ref.S

#    return F, M
#end

#@enum PropType CO=1 COUNTER=-1 COCOUNTER=0

"""
    MotorPropBatteryDataFit(CT2, CT1, CT0, CQ2, CQ1, CQ0, D, num, type,
        R, Kv, i0, voltage)

**Inputs**
- CT2, CT1, CT0: quadratic fit to propeller thrust coefficient of form: CT = CT2*J2 + CT1*J + CT0
- CQ2, CQ1, CQ0: quadratic fit to propeller torque coefficient of form: CQ = CQ2*J2 + CQ1*J + CQ0
- D: propeller diameter
- num: number of propellers
- type: CO (torques add), COUNTER (torques add but with minus sign), COCOUNTER (no torque, they cancel out)
- R: motor resistance
- Kv: motor Kv
- i0: motor no-load current
- voltage: battery voltage
"""
#struct MotorPropBatteryDataFit{TF, TI, PropType} <: AbstractPropulsionModel
    # CT = CT2*J2 + CT1*J + CT0
    # CQ = CQ2*J2 + CQ1*J + CQ0
#    CT2::TF  # prop data fit
#    CT1::TF
#    CT0::TF
#    CQ2::TF
#    CQ1::TF
#    CQ0::TF
#    D::TF  # prop diameter
#    num::TI
#    type::PropType

#    R::TF  # motor resistance
#    Kv::TF  # motor Kv
#    i0::TF  # motor no-load current

#    voltage::TF  # battery voltage
#end

function omegadot(rpm, omega, mp, l, k, b)

    I = [mp.Ixx   0.0   0.0;
         0.0     mp.Iyy 0.0;
         0.0      0.0  mp.Izz]
    tau = torque(rpm, l, b, k)
    #Inv = [1/I[1,1];1/I[2,2];1/I[3,3]]
    #prod = [tau[1]*Inv[1];tau[2]*Inv[2];tau[3]*Inv[3]]
    #diff = [((I[2,2]-I[3,3])/I[1,1])*omega[2]*omega[3];((I[3,3]-I[1,1])/I[2,2])*omega[1]*omega[3];
    #((I[1,1]-I[2,2])/I[3,3])*omega[1]*omega[2]]
    #omegadot = prod - diff
    omegadot = inv(I) .* (tau - cross(omega, I .* omega))
    #it seems there is a problem with cross function because they are float
return omegadot
end

    # density
#    rho, _ = properties(atm, state)

#    D = prop.D

    # determine torque for motor/prop match (quadratic equation)
#    a = rho*D^5/(2*pi)^2 * prop.CQ0
#    b = rho*D^4/(2*pi)*Va * prop.CQ1 + 1.0/(prop.R*prop.Kv)
#    c = rho*D^3*Va^2 * prop.CQ2 - control.throttle*prop.voltage/(prop.R*prop.Kv) + prop.i0/prop.Kv
#    Omega = (-b + sqrt(b^2 - 4*a*c))/(2*a)

    # advance ratio
#    n = Omega/(2*pi)
#    J = Va/(n*D)

    # thrust and torque
#    CT = prop.CT0 + prop.CT1*J + prop.CT2*J^2
#    CQ = prop.CQ0 + prop.CQ1*J + prop.CQ2*J^2

#    T = prop.num * CT * rho * n^2 * D^4
#    Q = prop.num * CQ * rho * n^2 * D^5 * Int(prop.type)

#    return [T, 0, 0], [Q, 0, 0]
#end

"""
    UniformGravitationalField()

Assumes center of mass and center of gravity are coincident.
"""
#struct UniformGravitationalField <: AbstractInertialModel end

function omega2thetadot(theta,omega)
    ct, cp = cos.([theta[2], theta[1]])
    st, sp = sin.([theta[2], theta[1]])

    C = [1.0  0.0  -st;
        0.0   cp  ct*sp
        0.0  -sp  ct*cp]

    thetadot = inv(C)*omega
    return thetadot
end

#    W = mp.m * gravity(atm, state)
#    ct, cp = cos.([state.theta, state.phi])
#    st, sp = sin.([state.theta, state.phi])

#    Fg = W*[-st, ct*sp, ct*cp]
#    Mg = [zero(W), zero(W), zero(W)]  # no gravitational moment

#    return Fg, Mg
#end


"""
    ConstantAtmosphere(Wi, Wb, rho, asound, g)

Constant atmospheric properties.
"""
struct Atmosphere{TF,TV}
    Wi::TV
    Wb::TV
    rho::TF
    asound::TF
    g::TF
end



"""
    ConstantController(de, dr, da, df, throttle)

Just a dummy controller that outputs constant control outputs the whole time.
"""
function plotsx(n, x, t)
    inspectdr()
    x1 = zeros(n)
    y1 = zeros(n)
    z1= zeros(n)
    for c in 1:n
    x1[c] = x[1,c]
    y1[c] = x[2,c]
    z1[c] = x[3,c]
    end
    p = Plots.plot(t, x1, label = "x" ,lw=3)
    Plots.plot!(p, t, y1, label = "y" )
    Plots.plot!(p, t, z1, label = "z" )
    Plots.xlabel!("t (s) ")
end

function plotstheta(n, theta, t)
    inspectdr()
    theta1 = zeros(n)
    theta2 = zeros(n)
    theta3= zeros(n)
    for c in 1:n
    theta1[c] = theta[1,c]
    theta2[c] = theta[2,c]
    theta3[c] = theta[3,c]
    end
    p = Plots.plot(t, theta1, label = "roll(phi)" ,lw=3)
    Plots.plot!(p, t, theta2, label = "pitch(theta)" )
    Plots.plot!(p, t, theta3, label = "yaw(qsi)" )
    Plots.xlabel!("t (s) ")
end

function plotsomega(n, omega, t)
    inspectdr()
    omega1 = zeros(n)
    omega2 = zeros(n)
    omega3= zeros(n)
    for c in 1:n
    omega1[c] = omega[1,c]
    omega2[c] = omega[2,c]
    omega3[c] = omega[3,c]
    end
    p = Plots.plot(t, omega1, label = "ω1" ,lw=3)
    Plots.plot!(p, t, omega2, label = "ω2" )
    Plots.plot!(p, t, omega3, label = "ω3" )
    Plots.xlabel!("t (s) ")
end

#    de::TF
#    dr::TF
#    da::TF
#    df::TF
#    throttle::TF
#end

#function setcontrol(controller::ConstantController, time, atm, state, mp, ref)
#    return Control(controller.de, controller.dr, controller.da, controller.df, controller.throttle)
#end
# --------------------------------------------------------


# ------------- main functions (public) --------------

"""
    sixdof!(ds, s, params, time)

dynamic and kinematic ODEs.  Follows format used in DifferentialEquations package.
- s = x, y, z, phi, theta, psi, u, v, w, p, q, r (same order as State)
- params = control, massproperties, reference, aeromodel, propmodel, inertialmodel, atmmodel
"""
#function sixdof!(ds, s, params, time)

#    x, y, z, phi, theta, psi, u, v, w, p, q, r = s
#    mp, ref, aeromodel, propmodel, inertialmodel, atmmodel, controller = params

    # ---- controller -------
#    state = State(s...)
    # -----------------------

    # --------- forces and moments ---------
    # aerodynamics
#    Fa, Ma = aeroforces(aeromodel, atmmodel, state, control, ref, mp)

    # propulsion
#    Fp, Mp = propulsionforces(propmodel, atmmodel, state, control, ref, mp)

    # weight
#    Fg, Mg = gravityforces(inertialmodel, atmmodel, state, control, ref, mp)

    # total forces and moments
#    F = Fa + Fp + Fg
#    M = Ma + Mp + Mg

    # --------------------------------------


    # ----- derivative of state --------
#    Vb = [u, v, w]
#    omegab = [p, q, r]

    # linear kinematics
#    Rib = inertialtobody(state)
#    rdot = Rib' * Vb

    # angular kinematics
#    phidot = p + (q*sin(phi) + r*cos(phi))*tan(theta)
#    thetadot = q*cos(phi) - r*sin(phi)
#    psidot = (q*sin(phi) + r*cos(phi))/cos(theta)

    # linear dynamics
#    vdot = F/mp.m - cross(omegab, Vb)

    # angular dynamics
#    I = [mp.Ixx -mp.Ixy -mp.Ixz;
#         -mp.Iyz mp.Iyy -mp.Iyz;
#         -mp.Ixz -mp.Iyz mp.Izz]
#    omegadot = I \ (M - cross(omegab, I*omegab))

    # -------------------------

    # TODO: if we need more efficiency we can avoid allocating and then assigning.
#    ds[1:3] = rdot
#    ds[4] = phidot
#    ds[5] = thetadot
#    ds[6] = psidot
#    ds[7:9] = vdot
#    ds[10:12] = omegadot
#end


end # module
