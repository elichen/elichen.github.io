// Initialize Firestore
const db = firebase.firestore();

// DOM Elements
const setupScreen = document.getElementById('setup-screen');
const homeScreen = document.getElementById('home-screen');
const profileForm = document.getElementById('profile-form');
const userNameDisplay = document.getElementById('user-name');
const userPhoneDisplay = document.getElementById('user-phone');
const markFreeBtn = document.getElementById('mark-free-btn');
const durationModal = document.getElementById('duration-modal');
const freeUsersList = document.getElementById('free-users-list');

// User state
let currentUser = null;

// Check if user exists in localStorage
function checkExistingUser() {
    const userId = localStorage.getItem('userId');
    const userName = localStorage.getItem('userName');
    const userPhone = localStorage.getItem('userPhone');

    if (userId && userName && userPhone) {
        currentUser = { id: userId, name: userName, phone: userPhone };
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
}

// Handle profile form submission
profileForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const name = document.getElementById('name').value.trim();
    const phone = document.getElementById('phone').value.trim();
    
    // Generate a new user ID
    const userId = generateUserId();
    
    try {
        // Save to Firestore
        await db.collection('users').doc(userId).set({
            name,
            phoneNumber: phone,
            isFree: false,
            expiresAt: null,
            updatedAt: firebase.firestore.FieldValue.serverTimestamp()
        });
        
        // Save to localStorage
        localStorage.setItem('userId', userId);
        localStorage.setItem('userName', name);
        localStorage.setItem('userPhone', phone);
        
        currentUser = { id: userId, name, phone };
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