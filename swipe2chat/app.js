// Initialize Firebase services
const db = firebase.firestore();
const auth = firebase.auth();

// DOM Elements
const setupScreen = document.getElementById('setup-screen');
const homeScreen = document.getElementById('home-screen');
const profileForm = document.getElementById('profile-form');
const googleSignInBtn = document.getElementById('google-signin');
const userNameDisplay = document.getElementById('user-name');
const userPhoneDisplay = document.getElementById('user-phone');
const markFreeBtn = document.getElementById('mark-free-btn');
const durationModal = document.getElementById('duration-modal');
const freeUsersList = document.getElementById('free-users-list');
const profileImage = document.getElementById('profile-image');

// User state
let currentUser = null;

// Set up Google Auth Provider
const googleProvider = new firebase.auth.GoogleAuthProvider();
googleProvider.addScope('profile');
googleProvider.addScope('email');

// Handle Google Sign-in
googleSignInBtn.addEventListener('click', async () => {
    try {
        const result = await auth.signInWithPopup(googleProvider);
        const user = result.user;
        
        // Get user profile data
        const name = user.displayName;
        const phone = user.phoneNumber || '';
        const photoURL = user.photoURL;
        
        // Pre-fill the profile form
        document.getElementById('name').value = name;
        document.getElementById('phone').value = phone;
        
        if (photoURL) {
            profileImage.src = photoURL;
            profileImage.classList.remove('hidden');
        }
        
        // Show the profile form to collect any missing information
        googleSignInBtn.classList.add('hidden');
        profileForm.classList.remove('hidden');
        
    } catch (error) {
        console.error('Error during Google sign-in:', error);
        alert('Error signing in with Google. Please try again.');
    }
});

// Check if user exists in localStorage and Firestore
async function checkExistingUser() {
    const userId = localStorage.getItem('userId');
    const userName = localStorage.getItem('userName');
    const userPhone = localStorage.getItem('userPhone');
    const userPhoto = localStorage.getItem('userPhoto');

    if (userId && userName) {
        currentUser = { 
            id: userId, 
            name: userName, 
            phone: userPhone,
            photoURL: userPhoto
        };
        showHomeScreen();
    } else {
        showSetupScreen();
    }
}

// Show/hide screens
function showHomeScreen() {
    setupScreen.classList.add('hidden');
    homeScreen.classList.remove('hidden');
    displayUserInfo();
    setupRealtimeListeners();
}

function showSetupScreen() {
    setupScreen.classList.remove('hidden');
    homeScreen.classList.add('hidden');
}

// Display user information
function displayUserInfo() {
    userNameDisplay.textContent = `Name: ${currentUser.name}`;
    userPhoneDisplay.textContent = `Phone: ${currentUser.phone}`;
    
    // Update profile photo in home screen if available
    const userInfoSection = document.querySelector('.user-info');
    if (currentUser.photoURL) {
        const img = document.createElement('img');
        img.src = currentUser.photoURL;
        img.alt = 'Profile Photo';
        img.style.width = '60px';
        img.style.height = '60px';
        img.style.borderRadius = '50%';
        img.style.marginBottom = '10px';
        userInfoSection.insertBefore(img, userInfoSection.firstChild);
    }
}

// Handle profile form submission
profileForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const name = document.getElementById('name').value.trim();
    const phone = document.getElementById('phone').value.trim();
    const photoURL = profileImage.classList.contains('hidden') ? null : profileImage.src;
    
    // Generate a new user ID or use the Google UID if available
    const userId = auth.currentUser ? auth.currentUser.uid : generateUserId();
    
    try {
        // Save to Firestore
        await db.collection('users').doc(userId).set({
            name,
            phoneNumber: phone,
            photoURL: photoURL,
            isFree: false,
            expiresAt: null,
            updatedAt: firebase.firestore.FieldValue.serverTimestamp()
        });
        
        // Save to localStorage
        localStorage.setItem('userId', userId);
        localStorage.setItem('userName', name);
        localStorage.setItem('userPhone', phone);
        if (photoURL) localStorage.setItem('userPhoto', photoURL);
        
        currentUser = { id: userId, name, phone, photoURL };
        showHomeScreen();
    } catch (error) {
        console.error('Error saving profile:', error);
        alert('Error saving profile. Please try again.');
    }
});

// Generate a random user ID
function generateUserId() {
    return 'user-' + Math.random().toString(36).substr(2, 9);
}

// Handle marking user as free
markFreeBtn.addEventListener('click', () => {
    durationModal.classList.remove('hidden');
});

// Handle duration selection
durationModal.addEventListener('click', async (e) => {
    if (e.target.hasAttribute('data-duration')) {
        const duration = parseInt(e.target.getAttribute('data-duration'));
        durationModal.classList.add('hidden');
        
        try {
            const expiresAt = new Date();
            expiresAt.setMinutes(expiresAt.getMinutes() + duration);
            
            await db.collection('users').doc(currentUser.id).update({
                isFree: true,
                expiresAt: expiresAt.getTime(),
                updatedAt: firebase.firestore.FieldValue.serverTimestamp()
            });
        } catch (error) {
            console.error('Error updating availability:', error);
            alert('Error updating availability. Please try again.');
        }
    }
});

// Setup realtime listeners for free users
function setupRealtimeListeners() {
    // Listen for free users
    db.collection('users')
        .where('isFree', '==', true)
        .onSnapshot((snapshot) => {
            const now = Date.now();
            freeUsersList.innerHTML = '';
            
            snapshot.forEach((doc) => {
                const userData = doc.data();
                
                // Skip if availability has expired
                if (userData.expiresAt < now) {
                    // Automatically update the expired status
                    db.collection('users').doc(doc.id).update({
                        isFree: false,
                        expiresAt: null
                    });
                    return;
                }
                
                // Skip current user
                if (doc.id === currentUser.id) return;
                
                const userCard = createUserCard(userData);
                freeUsersList.appendChild(userCard);
            });
        });
}

// Create a user card element
function createUserCard(userData) {
    const div = document.createElement('div');
    div.className = 'user-card';
    
    // Add profile photo if available
    if (userData.photoURL) {
        const img = document.createElement('img');
        img.src = userData.photoURL;
        img.alt = 'Profile Photo';
        img.style.width = '40px';
        img.style.height = '40px';
        img.style.borderRadius = '50%';
        img.style.marginRight = '10px';
        div.appendChild(img);
    }
    
    const nameSpan = document.createElement('span');
    nameSpan.className = 'name';
    nameSpan.textContent = userData.name;
    
    const callButton = document.createElement('button');
    callButton.className = 'call-button';
    callButton.textContent = 'Call';
    callButton.onclick = () => {
        window.location.href = `tel:${userData.phoneNumber}`;
    };
    
    div.appendChild(nameSpan);
    div.appendChild(callButton);
    
    return div;
}

// Check for expired availability periodically
setInterval(() => {
    if (currentUser) {
        db.collection('users').doc(currentUser.id).get().then((doc) => {
            const userData = doc.data();
            if (userData.isFree && userData.expiresAt < Date.now()) {
                db.collection('users').doc(currentUser.id).update({
                    isFree: false,
                    expiresAt: null
                });
            }
        });
    }
}, 60000); // Check every minute

// Initialize the app
checkExistingUser(); 