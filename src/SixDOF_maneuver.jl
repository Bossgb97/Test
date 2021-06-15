#=##############################################################################
# DESCRIPTION
    Types defining maneuvers of flight vehicles.

# AUTHORSHIP
  * Author    : Eduardo J. Alvarez
  * Email     : Edo.AlvarezR@gmail.com
  * Created   : Oct 2019
  * License   : MIT
=###############################################################################

################################################################################
# ABSTRACT MANEUVER TYPE
################################################################################

using LinearAlgebra: norm, cross
using Plots

"""
    `AbstractManeuver{N, M}`

`N` indicates the number of tilting systems in this maneuver, while and `M`
indicates the number of rotor systems.

Every implementation of `AbstractManeuver` must have the properties:

 * `angle::NTuple{N, Function}` where `angle[i](t)` returns the angle of the
        i-th tilting system at time `t` (t is nondimensionalized by the total
        time of the maneuver, from 0 to 1, beginning to end).
 * `RPM::NTuple{M, Function}` where `RPM[i](t)` returns the normalized RPM of
        the i-th rotor system at time `t`. This RPM values are normalized by the
        an arbitrary RPM value (usually RPM in hover or cruise).
"""
abstract type AbstractManeuver{N, M} end

##### FUNCTIONS REQUIRED IN IMPLEMENTATIONS ####################################
"""
    `calc_dV(maneuver::AbstractManeuver, vehicle::Vehicle, t, dt, ttot, Vref)`

Returns the change in velocity `dV=[dVx, dVy, dVz]` (m/s) of `vehicle` performing
`maneuver` at time `t` (s) after a time step `dt` (s).  `Vref` and `tot` are the
reference velocity and the total time at which this maneuver is being performed,
respectively. `dV` is in the global reference system.
"""
function calc_dV(self::AbstractManeuver, vehicle::AbstractVehicle, t::Real,
                                            dt::Real, ttot::Real, Vref::Real)
    error("$(typeof(self)) has no implementation yet!")
end

"""
    `calc_dw(maneuver::AbstractManeuver, vehicle::Vehicle, t, dt, Vref, ttot)`

Returns the change in angular velocity `dW=[dWx, dWy, dWz]` (about global
axes, in radians) of `vehicle` performing `maneuver` at time `t` (s) after a
time step `dt` (s). `ttot` is the total time at which this maneuver is to be
performed.
"""
function calc_dW(self::AbstractManeuver, vehicle::AbstractVehicle, t::Real,
                                                        dt::Real, ttot::Real)
    error("$(typeof(self)) has no implementation yet!")
end

##### COMMON FUNCTIONS  ########################################################
"""
    `get_ntltsys(self::AbstractManeuver)`
Return number of tilting systems.
"""
get_ntltsys(self::AbstractManeuver) = typeof(self).parameters[1]

"""
    `get_nrtrsys(self::AbstractManeuver)`
Return number of rotor systems.
"""
get_nrtrsys(self::AbstractManeuver) = typeof(self).parameters[2]

"""
    `get_angle(maneuver::AbstractManeuver, i::Int, t::Real)`

Returns the angle (in degrees) of the i-th tilting system at the non-dimensional
time t.
"""
function get_angle(self::AbstractManeuver, i::Int, t::Real)
    if i<=0 || i>get_ntltsys(self)
        error("Invalid tilting system #$i (max is $(get_ntltsys(self))).")
    end
    if t<0 || t>1
        @warn("Got non-dimensionalized time $(t).")
    end
    return self.angle[i](t)
end

"""
    `get_angles(maneuver::AbstractManeuver, t::Real)`

Returns the angle (in degrees) of every tilting systems at the non-dimensional
time t.
"""
get_angles(self::AbstractManeuver, t::Real) = Tuple(a(t) for a in self.angle)

"""
    `get_RPM(maneuver::AbstractManeuver, i::Int, t::Real)`

Returns the normalized RPM of the i-th rotor system at the non-dimensional time
t.
"""
function get_RPM(self::AbstractManeuver, i::Int, t::Real)
    if i<=0 || i>get_nrtrsys(self)
        error("Invalid rotor system #$i (max is $(get_nrtrsys(self))).")
    end
    if t<0 || t>1
        @warn("Got non-dimensionalized time $(t).")
    end
    return self.RPM[i](t)
end

"""
    `get_RPMs(maneuver::AbstractManeuver, t::Real)`

Returns the normalized RPM of every rotor systems at the non-dimensional time
t.
"""
get_RPMs(self::AbstractManeuver, t::Real) = Tuple(rpm(t) for rpm in self.RPM)


##### COMMON INTERNAL FUNCTIONS  ###############################################

##### END OF ABSTRACT MANEUVER #################################################










################################################################################
# KINEMATIC MANEUVER TYPE
################################################################################
"""
    `KinematicManeuver(angle, RPM, Vvehicle, anglevehicle)`

A vehicle maneuver where the kinematic are prescribed.

# ARGUMENTS
* `angle::NTuple{N, Function}` where `angle[i](t)` returns the angles
        `[Ax, Ay, Az]` (in degrees)of the i-th tilting system at time `t` (t is
        nondimensionalized by the total time of the maneuver, from 0 to 1,
        beginning to end).
* `RPM::NTuple{M, Function}` where `RPM[i](t)` returns the normalized RPM of
        the i-th rotor system at time `t`. This RPM values are normalized by the
        an arbitrary RPM value (usually RPM in hover or cruise).
* `Vvehicle::Function` where `Vvehicle(t)` returns the normalized vehicle
        velocity `[Vx, Vy, Vz]` at the normalized time `t`. Velocity is
        normalized by a reference velocity (typically, cruise velocity).
* `anglevehicle::Function` where `anglevehicle(t)` returns the angles
        `[Ax, Ay, Az]` (in degrees) of the vehicle relative to the global
        coordinate system at the normalized time `t`.
"""
struct KinematicManeuver{N, M} <: AbstractManeuver{N, M}
    angle::NTuple{N, Function}
    RPM::NTuple{M, Function}
    Vvehicle::Function
    anglevehicle::Function
end

# # Implicit N and M constructor
# KinematicManeuver(a::NTuple{N, Function}, b::NTuple{M, Function},
#                     c::Function, d::Function
#                  ) where {N, M} = KinematicManeuver{N, M}(a, b, c, d)


##### FUNCTIONS  ###############################################################
function calc_dV(self::KinematicManeuver, vehicle::AbstractVehicle, t::Real,
                                            dt::Real, ttot::Real, Vref::Real)
    return Vref * (self.Vvehicle((t+dt)/ttot) - self.Vvehicle(t/ttot))
end

function calc_dW(self::KinematicManeuver, vehicle::AbstractVehicle, t::Real,
                                                         dt::Real, ttot::Real)
    prev_W = (self.anglevehicle(t/ttot) - self.anglevehicle((t-dt)/ttot)) / dt
    cur_W = (self.anglevehicle((t+dt)/ttot) - self.anglevehicle(t/ttot)) / dt
    return pi/180 * (cur_W - prev_W)
end


##### INTERNAL FUNCTIONS  ######################################################

##### END OF KINEMATICMANEUVER  ################################################










################################################################################
# DYNAMIC MANEUVER TYPE
################################################################################
"""
    `DynamicManeuver(angle, RPM, Vvehicle, anglevehicle)`

A vehicle maneuver where the kinematic are prescribed.

# ARGUMENTS
* `angle::NTuple{N, Function}` where `angle[i](t)` returns the angles
        `[Ax, Ay, Az]` (in degrees)of the i-th tilting system at time `t` (t is
        nondimensionalized by the total time of the maneuver, from 0 to 1,
        beginning to end).
* `RPM::NTuple{M, Function}` where `RPM[i](t)` returns the normalized RPM of
        the i-th rotor system at time `t`. This RPM values are normalized by the
        an arbitrary RPM value (usually RPM in hover or cruise).
* `Vvehicle::Function` where `Vvehicle(t)` returns the normalized vehicle
        velocity `[Vx, Vy, Vz]` at the normalized time `t`. Velocity is
        normalized by a reference velocity (typically, cruise velocity).
* `anglevehicle::Function` where `anglevehicle(t)` returns the angles
        `[Ax, Ay, Az]` (in degrees) of the vehicle relative to the global
        coordinate system at the normalized time `t`.
"""
struct DynamicManeuver{N, M} <: AbstractManeuver{N, M}
    #angle::NTuple{N, Function}
    RPM::NTuple{M, Function}
    Vvehicle::Function
    anglevehicle::Function
    omega::Function
end

##### INTERNAL FUNCTIONS  ######################################################
function calc_dV(self::DynamicManeuver, vehicle::AbstractVehicle, t::Real,
                                            dt::Real, ttot::Real, Vref::Real, tinit::Real )
            c = (t - tinit)/dt
            c = trunc(Int,c)
    return  self.Vvehicle(c) - self.Vvehicle((c-1))
end

function calc_dW(self::DynanicManeuver, vehicle::AbstractVehicle, t::Real,
                                                         dt::Real, ttot::Real, tinit::Real)
    c = (t - tinit)/dt
    c = trunc(Int,c)
    ntot = (ttot-tinit)/dt
    ntot = trunc(Int,n)
    if c <= (ntot-1)
    prev_W = (self.omega(c) - self.omega((c-1))
    cur_W = (self.omega((c+1)) - self.omega(c)
    return pi/180 * (cur_W - prev_W)
    end
end
##### END OF KINEMATICMANEUVER  ################################################

#
