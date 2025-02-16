# Swipe2Chat

A mobile-friendly web application that lets users indicate their availability for a chat and easily connect with others who are free to talk.

## Features

- Simple profile creation with name and phone number
- Set your availability with customizable duration (15min, 30min, 1hour)
- See who else is currently free to chat
- One-tap calling through the phone dialer
- Automatic availability reset when time expires
- Real-time updates of user availability

## Setup

1. Create a Firebase project:
   - Go to [Firebase Console](https://console.firebase.google.com)
   - Create a new project
   - Enable Firestore Database
   - Set up Firestore security rules (see below)
   - Register a web app in your project
   - Copy the Firebase configuration

2. Update the Firebase configuration:
   - Open `config.js`
   - Replace the placeholder values with your Firebase configuration

3. Set up Firestore Security Rules:
   ```javascript
   rules_version = '2';
   service cloud.firestore {
     match /databases/{database}/documents {
       match /users/{userId} {
         allow read: if true;
         allow write: if true;
       }
     }
   }
   ```

4. Deploy the application:
   - Option 1: Deploy to Firebase Hosting
   - Option 2: Deploy to any static web hosting service (GitHub Pages, Netlify, etc.)

## Local Development

1. Clone the repository
2. Update Firebase configuration in `config.js`
3. Serve the files using a local web server
   ```bash
   # Using Python 3
   python -m http.server 8000
   
   # Or using Node.js http-server
   npx http-server
   ```
4. Open `http://localhost:8000` in your browser

## Technical Details

- Pure static web application (HTML, CSS, JavaScript)
- Firebase Firestore for real-time data storage
- Mobile-first responsive design
- No server-side code required
- Automatic cleanup of expired availability status

## Security Considerations

- The current implementation uses basic Firestore security rules for simplicity
- In a production environment, consider implementing:
  - Phone number verification
  - Rate limiting
  - More restrictive security rules
  - User authentication

## Future Enhancements

- Phone number verification
- Friend lists/groups
- Custom availability messages
- Profile pictures
- Call history
- Availability scheduling

## License

MIT License - feel free to use this code for your own projects! 