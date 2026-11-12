# Q2 budget review (engineering)

Attendees: Lena (finance), Tom, me.

## Cloud spend

- Q1 actual: $41,200, against a plan of $36,000. Most of the overrun is the three staging
  clusters that run around the clock.
- Q2 plan: $34,500. Staging clusters scale to zero outside working hours, saving roughly
  $5,000 per quarter. Move build artifacts older than 90 days to cold storage.
- Reserved instances for the two database servers: approved, pays back in 7 months.

## People

- Approved: two contractors for the mobile app beta, 3 months each.
- Not approved: the observability vendor upgrade. Revisit in Q3.

## Follow-ups

- Tom sends the tagging policy so every resource has a team and cost-center tag.
- Finance wants a monthly spend report instead of quarterly.
