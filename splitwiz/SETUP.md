# SplitWiz Setup Guide

This guide will help you set up a fresh Supabase project for SplitWiz.

## Prerequisites

- A Supabase account (sign up at https://supabase.com)
- A Google Cloud Console project for OAuth (https://console.cloud.google.com)

## Step 1: Create a New Supabase Project

1. Log in to Supabase Dashboard
2. Click "New Project"
3. Fill in:
   - Project name: `splitwiz` (or your choice)
   - Database Password: (save this securely)
   - Region: (choose closest to your users)
4. Click "Create Project" and wait for setup

## Step 2: Configure Authentication

### Enable Google OAuth:

1. Go to **Authentication** → **Providers**
2. Find **Google** and click to expand
3. Toggle **Enable Sign in with Google**
4. You'll need:
   - **Client ID** (from Google Cloud Console)
   - **Client Secret** (from Google Cloud Console)

### Set up Google OAuth:

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a new project or select existing
3. Enable Google+ API:
   - Go to **APIs & Services** → **Library**
   - Search for "Google+ API" and enable it
4. Create OAuth credentials:
   - Go to **APIs & Services** → **Credentials**
   - Click **Create Credentials** → **OAuth client ID**
   - Application type: **Web application**
   - Name: `SplitWiz`
   - Authorized JavaScript origins: 
     - `http://localhost:8000` (for development)
     - Your production domain
   - Authorized redirect URIs:
     - `https://[YOUR-PROJECT-REF].supabase.co/auth/v1/callback`
     - Find your project ref in Supabase project settings
5. Copy the Client ID and Client Secret to Supabase

### Configure Auth Settings:

1. In Supabase, go to **Authentication** → **URL Configuration**
2. Add your site URL (e.g., `https://yourdomain.com/splitwiz`)
3. Add redirect URLs:
   - `http://localhost:8000/splitwiz` (for development)
   - `https://yourdomain.com/splitwiz` (for production)

## Step 3: Set Up Database

1. Go to **SQL Editor** in Supabase
2. Create a new query
3. Copy and paste the entire contents of `sql.txt`
4. Click **Run** to execute
5. You should see success messages for all table creations

## Step 4: Configure the Application

1. In Supabase, go to **Settings** → **API**
2. Copy:
   - **Project URL** (looks like `https://[YOUR-PROJECT-REF].supabase.co`)
   - **Anon/Public Key** (safe for client-side use)
3. Update `app.js`:
   ```javascript
   const SUPABASE_URL = 'your-project-url-here';
   const SUPABASE_ANON_KEY = 'your-anon-key-here';
   ```

## Step 5: How Invitations Work

SplitWiz uses a simple invite link system:

1. **Share invite links**: Each trip has a unique invite link that can be shared via any messaging app
2. **Automatic joining**: When someone clicks the link and signs in with Google, they're automatically added to the trip
3. **No email setup required**: The app works without any email configuration

## Step 6: Test Your Setup

1. Serve the app locally:
   ```bash
   cd splitwiz
   python -m http.server 8000
   # or
   npx http-server
   ```
2. Navigate to `http://localhost:8000`
3. Test:
   - Sign in with Google
   - Create a trip
   - Click "Manage Participants" to get the invite link
   - Share the link with friends
   - Add expenses and see balances update

## Step 7: Deploy to Production

1. Host the static files on your preferred platform:
   - GitHub Pages (update repository settings)
   - Netlify (drag and drop the folder)
   - Vercel (connect GitHub repo)
   - Any static hosting service

2. Update the redirect URLs in:
   - Supabase Authentication settings
   - Google Cloud Console OAuth settings

3. Update `app.js` if using a custom domain

## Troubleshooting

### "Infinite recursion" error
- Run the SQL script again to ensure all policies are updated
- Check that you're using the latest version of `sql.txt`

### Google login redirects but doesn't log in
- Verify redirect URLs match exactly in Google Console and Supabase
- Check browser console for errors
- Ensure cookies are enabled

### Friends can't join trips
- Make sure they sign in with Google first
- Check that the invite link includes the trip ID parameter
- Verify they have permission to access the app URL

## Security Notes

- Never expose your `service_role` key (only use `anon` key in client)
- The RLS policies ensure users can only:
  - See trips they're part of
  - Edit/delete their own expenses
  - Add participants to trips they're in
- All data is filtered at the database level for security