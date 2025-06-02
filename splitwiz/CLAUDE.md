# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SplitWiz is a client-side expense splitting web application using vanilla JavaScript with Supabase backend. It allows users to create trips, add expenses, and track who owes whom.

## Commands

### Local Development
```bash
cd splitwiz
python -m http.server 8000  # or use any static server
# Navigate to http://localhost:8000
```

### Database Setup
1. Create a new Supabase project
2. Run the contents of `sql.txt` in Supabase SQL editor
3. Configure Google OAuth in Supabase Authentication settings
4. Update `app.js` with your Supabase URL and anon key

## Architecture

### Frontend Structure
- **Single Page Application** - All UI in `index.html` with JavaScript-controlled visibility
- **Modal-based UI** - Forms for creating trips, adding expenses, managing participants
- **Toast Notifications** - Custom notification system replacing browser alerts
- **No Build Process** - Vanilla JS, can be deployed as static files

### Data Flow
1. **Authentication**: Google OAuth via Supabase → User session stored in memory
2. **Trip Access**: User joins trip via invite link → Automatic participant record creation
3. **Expense Tracking**: User adds expense → Split calculations stored in database
4. **Balance Calculation**: Real-time calculation from expenses and splits

### Database Schema (RLS-protected)
- `trips` → `trip_participants` → `expenses` → `expense_splits`
- All tables use UUID primary keys and have RLS policies
- Cascade deletion ensures data integrity
- Custom `user_profiles` view exposes safe user data

### Key Supabase Integration Points
```javascript
// Supabase client initialization (app.js:2-4)
const SUPABASE_URL = 'your-project-url';
const SUPABASE_ANON_KEY = 'your-anon-key';
const supabase = window.supabase.createClient(SUPABASE_URL, SUPABASE_ANON_KEY);
```

### Security Model
- Row Level Security (RLS) enforces access control at database level
- Users can only see/modify trips they're participants in
- Only expense owners can edit/delete their expenses
- Trip owners have additional permissions (delete trip, remove participants)

## Important Implementation Details

### Invite System
- No email sending - uses shareable links: `https://domain.com/splitwiz?trip={tripId}`
- Clicking invite link automatically adds user as participant
- No separate invitation tracking table

### Balance Calculations
- Performed client-side from expense data
- Shows "Nothing owed" when balanced (not "settled up" since app doesn't track payments)
- Balances displayed as "X owes $Y" or "X gets back $Y"

### Modal State Management
- Modals controlled via CSS classes (`.active`)
- Global functions exposed via `window` object for onclick handlers
- Form submissions prevent default and handle async operations

### Common Issues & Solutions
1. **RLS Recursion**: Policies simplified to avoid circular references
2. **Toast Container Visibility**: Uses `display: none` + dynamic class toggling
3. **Login Screen Centering**: Requires `display: flex !important` on `.active` state