// Initialize Firebase services
const db = firebase.firestore();
const auth = firebase.auth();

// DOM Elements
const setupScreen = document.getElementById('setup-screen');
const homeScreen = document.getElementById('home-screen');
const profileForm = document.getElementById('profile-form');
const quickSignInForm = document.getElementById('quick-signin-form');
const googleSignInBtn = document.getElementById('google-signin');
const userNameDisplay = document.getElementById('user-name');
const userPhoneDisplay = document.getElementById('user-phone');
const profileImage = document.getElementById('profile-image');
const durationModal = document.getElementById('duration-modal');
const cardStack = document.getElementById('card-stack');
const noUsersMessage = document.querySelector('.no-users-message');
const matchModal = document.getElementById('match-modal');
const matchUser1Photo = document.getElementById('match-user1-photo');
const matchUser2Photo = document.getElementById('match-user2-photo');
const matchName = document.getElementById('match-name');
const callMatchBtn = document.getElementById('call-match-btn');
const closeMatchBtn = document.getElementById('close-match-btn');
const swipeLeftBtn = document.getElementById('swipe-left');
const swipeRightBtn = document.getElementById('swipe-right');

// User state
let currentUser = null;
let currentCard = null;
let currentCardData = null;
let swipingEnabled = true;
let swipedUserIds = []; // added to track already swiped profiles
let lastFreeUsers = []; // store the latest snapshot of free users

// Set up Google Auth Provider
const googleProvider = new firebase.auth.GoogleAuthProvider();
googleProvider.addScope('profile');
googleProvider.addScope('email');

// Handle quick sign-in
quickSignInForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const name = document.getElementById('quick-name').value.trim();
    const phone = document.getElementById('quick-phone').value.trim();
    const userId = generateUserId();
    
    try {
        // Save to Firestore
        await db.collection('users').doc(userId).set({
            name,
            phoneNumber: phone,
            photoURL: null,
            isFree: false,
            expiresAt: null,
            updatedAt: firebase.firestore.FieldValue.serverTimestamp(),
            isQuickSignIn: true
        });
        
        // Save to localStorage
        localStorage.setItem('userId', userId);
        localStorage.setItem('userName', name);
        localStorage.setItem('userPhone', phone);
        localStorage.setItem('isQuickSignIn', 'true');
        
        currentUser = { 
            id: userId, 
            name, 
            phone: phone,
            photoURL: null,
            isQuickSignIn: true
        };
        
        showDurationPrompt();
    } catch (error) {
        console.error('Error during quick sign-in:', error);
        alert('Error signing in. Please try again.');
    }
});

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
        
        // Use the global profileImage element and update if available
        if (photoURL && profileImage) {
            profileImage.src = photoURL;
            profileImage.classList.remove('hidden');
        }
        
        // Show the profile form to collect any missing information
        googleSignInBtn.classList.add('hidden');
        quickSignInForm.classList.add('hidden');
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
    const isQuickSignIn = localStorage.getItem('isQuickSignIn') === 'true';

    if (userId && userName) {
        currentUser = { 
            id: userId, 
            name: userName, 
            phone: userPhone,
            photoURL: userPhoto,
            isQuickSignIn
        };
        showDurationPrompt();
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
    
    // Only show phone if it exists
    if (currentUser.phone) {
        userPhoneDisplay.textContent = `Phone: ${currentUser.phone}`;
        userPhoneDisplay.classList.remove('hidden');
    } else {
        userPhoneDisplay.classList.add('hidden');
    }
    
    // Update profile photo in home screen with fallback
    const userInfoSection = document.querySelector('.user-info');
    // Remove any existing image to avoid duplicates
    const existingImg = userInfoSection.querySelector('img');
    if(existingImg) {
        existingImg.remove();
    }
    const img = document.createElement('img');
    img.src = currentUser.photoURL || `https://ui-avatars.com/api/?name=${encodeURIComponent(currentUser.name || 'User')}`;
    img.alt = 'Profile Photo';
    img.style.width = '60px';
    img.style.height = '60px';
    img.style.borderRadius = '50%';
    img.style.marginBottom = '10px';
    userInfoSection.insertBefore(img, userInfoSection.firstChild);
}

// Handle profile form submission
profileForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const name = document.getElementById('name').value.trim();
    const phone = document.getElementById('phone').value.trim();
    const profileImage = document.getElementById('profile-image');
    const photoURL = profileImage && !profileImage.classList.contains('hidden') ? profileImage.src : null;
    
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
        showDurationPrompt();
    } catch (error) {
        console.error('Error saving profile:', error);
        alert('Error saving profile. Please try again.');
    }
});

// Generate a random user ID
function generateUserId() {
    return 'user-' + Math.random().toString(36).substr(2, 9);
}

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
            
            // Show time remaining
            const timeDisplay = document.createElement('div');
            timeDisplay.className = 'time-remaining';
            timeDisplay.textContent = `Free for ${duration} minutes`;
            document.querySelector('.availability-controls').appendChild(timeDisplay);
            
            // Show and start the swipe interface
            const swipeContainer = document.querySelector('.swipe-container');
            swipeContainer.classList.remove('hidden');
            setupRealtimeListeners();
            
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
            const freeUsers = [];
            
            snapshot.forEach((doc) => {
                const userData = doc.data();
                
                // Skip if availability has expired
                if (userData.expiresAt < now) {
                    db.collection('users').doc(doc.id).update({
                        isFree: false,
                        expiresAt: null
                    });
                    return;
                }
                
                // Skip current user
                if (doc.id === currentUser.id) return;
                
                freeUsers.push({ id: doc.id, ...userData });
            });
            
            updateCardStack(freeUsers);
        });
}

function updateCardStack(users) {
    console.log('updateCardStack called with users:', users);
    lastFreeUsers = users; // store the latest snapshot
    cardStack.innerHTML = '';
    // Filter out users that have already been swiped
    const availableUsers = users.filter(user => !swipedUserIds.includes(user.id));
    console.log('Available users count:', availableUsers.length);
    if (availableUsers.length === 0) {
        console.log('No available users found, showing no-users-message');
        noUsersMessage.classList.remove('hidden');
        return;
    }
    console.log('Available users exist, hiding no-users-message');
    noUsersMessage.classList.add('hidden');
    createNewCard(availableUsers[0]);
}

function createNewCard(userData) {
    const card = document.createElement('div');
    card.className = 'card sliding-in';
    
    const photo = document.createElement('img');
    photo.className = 'card-photo';
    photo.src = userData.photoURL || 'https://ui-avatars.com/api/?name=' + encodeURIComponent(userData.name);
    photo.alt = userData.name;
    
    const info = document.createElement('div');
    info.className = 'card-info';
    
    const name = document.createElement('div');
    name.className = 'card-name';
    name.textContent = userData.name;
    
    const status = document.createElement('div');
    status.className = 'card-status';
    status.textContent = 'Free to chat!';
    
    info.appendChild(name);
    info.appendChild(status);
    card.appendChild(photo);
    card.appendChild(info);
    
    cardStack.appendChild(card);
    currentCard = card;
    currentCardData = userData;
    
    initializeSwipe(card);
}

function initializeSwipe(card) {
    let startX = 0;
    let currentX = 0;
    let isDragging = false;
    
    card.addEventListener('mousedown', handleDragStart);
    card.addEventListener('mousemove', handleDragMove);
    card.addEventListener('mouseup', handleDragEnd);
    card.addEventListener('mouseleave', handleDragEnd);
    
    card.addEventListener('touchstart', (e) => handleDragStart(e.touches[0]));
    card.addEventListener('touchmove', (e) => handleDragMove(e.touches[0]));
    card.addEventListener('touchend', handleDragEnd);
    
    function handleDragStart(e) {
        if (!swipingEnabled) return;
        isDragging = true;
        startX = e.clientX;
        card.classList.add('swiping');
    }
    
    function handleDragMove(e) {
        if (!isDragging || !swipingEnabled) return;
        currentX = e.clientX - startX;
        const rotation = currentX * 0.1;
        card.style.transform = `translateX(${currentX}px) rotate(${rotation}deg)`;
    }
    
    function handleDragEnd() {
        if (!isDragging || !swipingEnabled) return;
        isDragging = false;
        card.classList.remove('swiping');
        
        if (Math.abs(currentX) > 100) {
            const direction = currentX > 0 ? 'right' : 'left';
            completeSwipe(direction);
        } else {
            card.style.transform = '';
        }
    }
}

async function completeSwipe(direction) {
    if (!currentCardData) return;
    // Record this swiped user so that they don't show up again
    swipedUserIds.push(currentCardData.id);
    swipingEnabled = false;
    currentCard.style.transition = 'transform 0.5s';
    currentCard.style.transform = `translateX(${direction === 'right' ? '150%' : '-150%'})`;
    
    if (direction === 'right') {
        // Directly show call modal without mutual match checking
        showCallModal(currentCardData);
    }
    
    // Remove the card after animation and show next card if available
    setTimeout(() => {
        currentCard.remove();
        currentCard = null;
        currentCardData = null;
        swipingEnabled = true;

        // Check for next available user
        const availableUsers = lastFreeUsers.filter(user => !swipedUserIds.includes(user.id));
        console.log('Looking for next card, available users:', availableUsers.length);
        if (availableUsers.length > 0) {
            console.log('Creating next card');
            createNewCard(availableUsers[0]);
            noUsersMessage.classList.add('hidden');
        } else {
            console.log('No more users available, showing no-users-message');
            noUsersMessage.classList.remove('hidden');
        }
    }, 500);
}

// New function to show call confirmation modal
function showCallModal(targetUser) {
    // Repurpose the modal to confirm calling the chosen person
    // Hide the current user's photo
    matchUser1Photo.style.display = 'none';
    matchUser2Photo.style.display = 'block';
    
    // Update modal title and message
    const modalTitle = matchModal.querySelector('h2');
    modalTitle.textContent = 'Call Now?';
    const modalMessage = matchModal.querySelector('p');
    modalMessage.textContent = `Would you like to call ${targetUser.name || 'this person'}?`;
    
    // Set the target user's photo with fallback
    matchUser2Photo.src = targetUser.photoURL || `https://ui-avatars.com/api/?name=${encodeURIComponent(targetUser.name || 'User')}`;
    
    // Update the displayed name
    matchName.textContent = targetUser.name || 'User';
    
    // Set the call button action
    const targetPhone = targetUser.phoneNumber || '';
    callMatchBtn.onclick = () => {
        window.location.href = `tel:${targetPhone}`;
    };
    
    // Show the modal
    matchModal.classList.remove('hidden');
}

// Handle swipe buttons
swipeLeftBtn.addEventListener('click', () => {
    if (currentCard && swipingEnabled) {
        completeSwipe('left');
    }
});

swipeRightBtn.addEventListener('click', () => {
    if (currentCard && swipingEnabled) {
        completeSwipe('right');
    }
});

closeMatchBtn.addEventListener('click', () => {
    console.log('closeMatchBtn clicked, cardStack children count:', cardStack.children.length);
    matchModal.classList.add('hidden');
    // Compute available users from the latest snapshot
    const availableUsers = lastFreeUsers.filter(user => !swipedUserIds.includes(user.id));
    console.log('Available users from last snapshot after swipe:', availableUsers.length);
    if (availableUsers.length === 0) {
        console.log('No available users remaining, showing no-users-message');
        noUsersMessage.classList.remove('hidden');
    } else {
        // If there is no card currently displaying, create a new card
        if (cardStack.children.length === 0) {
            console.log('Creating new card for available user:', availableUsers[0]);
            createNewCard(availableUsers[0]);
        }
    }
});

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

// Show duration prompt immediately after sign-in
function showDurationPrompt() {
    setupScreen.classList.add('hidden');
    homeScreen.classList.remove('hidden');
    
    // Hide swipe container initially
    document.querySelector('.swipe-container').classList.add('hidden');
    
    // Show duration modal immediately
    durationModal.classList.remove('hidden');
    
    displayUserInfo();
}

// Initialize the app
checkExistingUser(); 