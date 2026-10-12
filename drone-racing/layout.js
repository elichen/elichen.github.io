// Where the pilot stands for a course: behind the pad, looking across the course,
// like an RC pilot at the flight line. The photo field is centered on the pilot.
export const EYE_HEIGHT = 1.6;   // m, the panorama's camera height

export function pilotSpot(flight) {
    const c = flight.course;
    const center = [0, 0];
    for (let i = 0; i < c.count; i++) { center[0] += c.centers[i][0] / c.count; center[1] += c.centers[i][1] / c.count; }
    const away = [flight.pad[0] - center[0], flight.pad[1] - center[1]];
    const n = Math.hypot(...away);
    // Far enough back that the whole course fits in view
    let reach = 0;
    for (let i = 0; i < c.count; i++) reach = Math.max(reach, Math.hypot(c.centers[i][0] - center[0], c.centers[i][1] - center[1]));
    const back = Math.max(n + 5, reach * 1.25 + 4);
    const pilot = [center[0] + away[0] / n * back, center[1] + away[1] / n * back];
    return { pilot, center, heading: Math.atan2(-away[1], -away[0]) };
}
