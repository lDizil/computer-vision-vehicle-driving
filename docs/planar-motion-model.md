# Planar braking model

This experimental point-mass model extrapolates a co-associated snapshot. It is
not a tyre, steering, obstacle or actuator safety model. The default fallback
remains the longitudinal model until explicitly selected by a composition root.

## Observation boundary

The OpenScene sidecar retains the four consecutive past/current CAN records,
raw orientations and body angular velocities. It binds them to the immutable
Fast bundle and pinned metadata SHA256; it never rewrites legacy evidence.
The wire identity contains raw source values and method identifiers only.
Derived floating-point diagnostics are recomputed on the reading platform,
so last-bit differences between Python/libm versions do not change identity.
Frame timestamps identify LiDAR associations. Pose/motion capture timestamps
and uncertainty are not retained by the producer and remain unknown.

Associated planar heading rate is the derivative of the pinned NAVSIM
`Quaternion.yaw_pitch_roll[0]` convention under `q_dot = q * (0, omega_body)/2`.
It is not generally body gyro Z. Previous pose differences are interval means,
not an instantaneous rate or a confidence bound. Singular headings are absent,
not replaced by zero. No future pose, annotation or evaluator output is used.

Source coordinate semantics: [pinned nuPlan schema](https://github.com/motional/nuplan-devkit/blob/ce3c323af01c0d7ec5672f7832ef53f9c679aab0/docs/nuplan_schema.md#L39-L47). OpenScene copies the local angular-rate triplet into CAN fields 13–15; its metadata producer does not retain the pose measurement timestamp.

## Independent path and body heading

For initial velocity `v0`, acceleration `a0`, speed `V0 > 0`, tangent `T0=v0/V0`
and left normal `N0`, define:

```
a_t0 = dot(T0, a0)
a_n0 = dot(N0, a0)
kappa = a_n0 / V0^2
rho = associated_heading_rate / V0
beta0 = atan2(vy, vx)
```

A nonnegative scalar jerk-limited speed profile supplies arc length `s(t)`.
Position follows a constant-curvature arc with initial tangent `T0`:

```
p(s) = T0 * sin(kappa*s)/kappa + N0 * (1-cos(kappa*s))/kappa
theta(s) = rho*s
beta(s) = beta0 + (kappa-rho)*s
```

The zero-curvature limit is analytic; the implementation uses stable sinc
forms rather than subtracting nearly equal cosines. Body heading begins at
zero, independently of velocity direction. This preserves the supplied planar
velocity, acceleration and associated heading derivative under the stated
snapshot hypothesis. It deliberately permits prescribed variable sideslip,
not a no-slip bicycle model. Reverse/pure-lateral initial motion is outside
this forward model. Unsupported observations are not rounded to rest.

## Continuous sufficient bounds

For scalar bounds `0 <= v <= V`, `|a_t| <= B`, `|j_t| <= q`, all phases including
those beyond the sampled output horizon satisfy:

```
p_ddot  = a_t*T + kappa*v^2*N
p_dddot = (j_t-kappa^2*v^3)*T + 3*kappa*v*a_t*N
|p_ddot|  <= hypot(B, abs(kappa)*V^2)
|p_dddot| <= hypot(q+kappa^2*V^3, 3*abs(kappa)*V*B)
|theta_dot| <= abs(rho)*V
```

These are conservative upper bounds, not measured maxima. The full-vector
acceleration and jerk limits retain their original meaning; they are not
silently reinterpreted as tangential limits. A bounded budget search may fail
to find a certificate even when another profile exists. Such failure is
unproven admission, not proof of global physical infeasibility.

A stopping distance denotes arc length, not longitudinal displacement.
Heading sample increments must remain unambiguous, and forward body motion
requires the complete unwrapped sideslip range to stay within `(-pi/2, pi/2)`.
The sampled kinematic checker is an additional numerical check, not a substitute
for these continuous bounds or a road-safety guarantee.
