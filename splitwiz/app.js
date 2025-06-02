// Initialize Supabase client
const SUPABASE_URL = 'https://vvxklqonwtybxqpoydqn.supabase.co';
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZ2eGtscW9ud3R5YnhxcG95ZHFuIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDg4MTkzNjUsImV4cCI6MjA2NDM5NTM2NX0.ff1eqGnqBeu83NIIFM1mgkybkHoQm5ngsk_7M1DMYT0';
const supabase = window.supabase.createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

// Toast notification system
function showToast(message, type = 'info', title = '') {
    const toastContainer = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    const icons = {
        success: '✓',
        error: '✕',
        info: 'ℹ'
    };
    
    const titles = {
        success: 'Success',
        error: 'Error',
        info: 'Info'
    };
    
    toast.innerHTML = `
        <span class="toast-icon">${icons[type]}</span>
        <div class="toast-content">
            ${title || titles[type] ? `<div class="toast-title">${title || titles[type]}</div>` : ''}
            <div class="toast-message">${message}</div>
        </div>
        <button class="toast-close" onclick="removeToast(this)">×</button>
    `;
    
    toastContainer.appendChild(toast);
    toastContainer.classList.add('has-toasts');
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        removeToast(toast.querySelector('.toast-close'));
    }, 5000);
}

function removeToast(closeBtn) {
    const toast = closeBtn.closest('.toast');
    const toastContainer = document.getElementById('toast-container');
    toast.classList.add('removing');
    setTimeout(() => {
        toast.remove();
        // Remove has-toasts class if no more toasts
        if (toastContainer.children.length === 0) {
            toastContainer.classList.remove('has-toasts');
        }
    }, 300);
}

window.removeToast = removeToast;

// Global state
let currentUser = null;
let currentTrip = null;
let participants = [];

// DOM Elements
const loginScreen = document.getElementById('login-screen');
const mainApp = document.getElementById('main-app');
const loadingOverlay = document.getElementById('loading-overlay');
const userEmail = document.getElementById('user-email');
const tripListView = document.getElementById('trip-list-view');
const tripDetailView = document.getElementById('trip-detail-view');
const tripsContainer = document.getElementById('trips-container');
const expensesContainer = document.getElementById('expenses-container');
const balancesContainer = document.getElementById('balances-container');
const tripNameElement = document.getElementById('trip-name');

// Auth Functions
async function signInWithGoogle() {
    try {
        const { data, error } = await supabase.auth.signInWithOAuth({
            provider: 'google',
            options: {
                redirectTo: window.location.origin + window.location.pathname
            }
        });
        if (error) throw error;
    } catch (error) {
        showToast('Error signing in: ' + error.message, 'error');
    }
}

async function signOut() {
    try {
        const { error } = await supabase.auth.signOut();
        if (error) throw error;
        window.location.reload();
    } catch (error) {
        showToast('Error signing out: ' + error.message, 'error');
    }
}

// UI Functions
function showLoading() {
    loadingOverlay.classList.add('active');
}

function hideLoading() {
    loadingOverlay.classList.remove('active');
}

function showModal(modalId) {
    document.getElementById(modalId).classList.add('active');
}

function closeModal(modalId) {
    document.getElementById(modalId).classList.remove('active');
}

window.closeModal = closeModal; // Make it globally accessible

function showTripList() {
    tripListView.classList.add('active');
    tripDetailView.classList.remove('active');
    loadTrips();
}

function showTripDetail(tripId) {
    currentTrip = tripId;
    tripListView.classList.remove('active');
    tripDetailView.classList.add('active');
    loadTripDetails();
}

// Trip Functions
async function loadTrips() {
    showLoading();
    try {
        // First get all trips where user is a participant
        const { data: participations, error: participationError } = await supabase
            .from('trip_participants')
            .select('trip_id')
            .eq('user_id', currentUser.id);

        if (participationError) throw participationError;

        if (!participations || participations.length === 0) {
            tripsContainer.innerHTML = '<p class="empty-state">No trips yet. Create your first trip!</p>';
            hideLoading();
            return;
        }

        const tripIds = participations.map(p => p.trip_id);

        // Now get the trip details
        const { data: trips, error: tripsError } = await supabase
            .from('trips')
            .select('*')
            .in('id', tripIds)
            .order('created_at', { ascending: false });

        if (tripsError) throw tripsError;

        if (!trips || trips.length === 0) {
            tripsContainer.innerHTML = '<p class="empty-state">No trips yet. Create your first trip!</p>';
            hideLoading();
            return;
        }

        // Get participant counts and balances for each trip
        const tripCards = await Promise.all(trips.map(async (trip) => {
            // Get participant count
            const { data: participants, error: pError } = await supabase
                .from('trip_participants')
                .select('user_id')
                .eq('trip_id', trip.id);

            const participantCount = pError ? 1 : (participants?.length || 1);

            // Calculate user's balance
            const balance = await calculateUserBalance(trip.id, currentUser.id);

            return createTripCard(trip, participantCount, balance);
        }));

        tripsContainer.innerHTML = tripCards.join('');
    } catch (error) {
        showToast('Error loading trips: ' + error.message, 'error');
    }
    hideLoading();
}

function createTripCard(trip, participantCount, balance) {
    const balanceClass = balance > 0 ? 'positive' : balance < 0 ? 'negative' : '';
    const balanceText = balance > 0 ? `You're owed $${balance.toFixed(2)}` : 
                       balance < 0 ? `You owe $${Math.abs(balance).toFixed(2)}` : 
                       'Nothing owed';

    return `
        <div class="trip-card" onclick="showTripDetail('${trip.id}')">
            <h3>${trip.name}</h3>
            <p class="participants">${participantCount} participants</p>
            <p class="balance ${balanceClass}">${balanceText}</p>
        </div>
    `;
}

async function createTrip(name) {
    showLoading();
    try {
        // Create trip
        const { data: trip, error: tripError } = await supabase
            .from('trips')
            .insert({
                name: name,
                created_by: currentUser.id
            })
            .select()
            .single();

        if (tripError) throw tripError;

        // Add creator as participant
        const { error: participantError } = await supabase
            .from('trip_participants')
            .insert({
                trip_id: trip.id,
                user_id: currentUser.id
            });

        if (participantError) throw participantError;

        closeModal('new-trip-modal');
        showToast('Trip created successfully!', 'success');
        showTripDetail(trip.id);
    } catch (error) {
        showToast('Error creating trip: ' + error.message, 'error');
    }
    hideLoading();
}

async function loadTripDetails() {
    showLoading();
    try {
        // Get trip info
        const { data: trip, error: tripError } = await supabase
            .from('trips')
            .select('*')
            .eq('id', currentTrip)
            .single();

        if (tripError) throw tripError;

        tripNameElement.textContent = trip.name;
        
        // Store current trip data for later use
        window.currentTripData = trip;
        
        // Show/hide delete button based on ownership
        const deleteBtn = document.getElementById('delete-trip-btn');
        if (trip.created_by === currentUser.id) {
            deleteBtn.style.display = 'inline-flex';
        } else {
            deleteBtn.style.display = 'none';
        }

        // Load participants
        await loadParticipants();

        // Load expenses
        await loadExpenses();

        // Calculate and show balances
        await calculateAndShowBalances();
    } catch (error) {
        showToast('Error loading trip details: ' + error.message, 'error');
    }
    hideLoading();
}

async function loadParticipants() {
    try {
        // Get all participants for this trip
        const { data: participantRecords, error } = await supabase
            .from('trip_participants')
            .select('user_id')
            .eq('trip_id', currentTrip);

        if (error) throw error;

        // Get user profiles for each participant
        const participantIds = participantRecords.map(p => p.user_id);
        
        const { data: profiles, error: profileError } = await supabase
            .from('user_profiles')
            .select('id, email, name, avatar_url')
            .in('id', participantIds);

        if (profileError) {
            console.error('Error loading profiles:', profileError);
            // Fallback: use basic info
            participants = participantRecords.map(p => ({
                id: p.user_id,
                email: 'User',
                name: 'User'
            }));
            return;
        }

        participants = profiles.map(p => ({
            id: p.id,
            email: p.email,
            name: p.name || p.email
        }));
    } catch (error) {
        console.error('Error loading participants:', error);
        // Ensure at least current user is in the list
        participants = [{
            id: currentUser.id,
            email: currentUser.email,
            name: currentUser.user_metadata?.full_name || currentUser.email
        }];
    }
}

async function loadExpenses() {
    try {
        const { data: expenses, error } = await supabase
            .from('expenses')
            .select(`
                *,
                user_profiles!paid_by(email, name),
                expense_splits(user_id, amount)
            `)
            .eq('trip_id', currentTrip)
            .order('expense_date', { ascending: false })
            .order('created_at', { ascending: false });

        if (error) throw error;

        if (!expenses || expenses.length === 0) {
            expensesContainer.innerHTML = '<p class="empty-state">No expenses yet</p>';
            return;
        }

        const expenseCards = expenses.map(expense => {
            const paidBy = expense.user_profiles.name || expense.user_profiles.email;
            const isOwner = expense.paid_by === currentUser.id;
            const splitUsers = expense.expense_splits.map(split => {
                const participant = participants.find(p => p.id === split.user_id);
                return participant ? participant.name : 'Unknown';
            }).join(', ');
            
            // Format date
            const expenseDate = new Date(expense.expense_date);
            const dateStr = expenseDate.toLocaleDateString('en-US', { 
                month: 'short', 
                day: 'numeric', 
                year: 'numeric' 
            });

            return `
                <div class="expense-item" data-expense='${JSON.stringify({
                    id: expense.id,
                    description: expense.description,
                    amount: expense.amount,
                    expense_date: expense.expense_date,
                    splits: expense.expense_splits.map(s => s.user_id)
                })}'>
                    <div class="expense-info">
                        <h4>${expense.description}</h4>
                        <p class="paid-by">Paid by ${paidBy} on ${dateStr}</p>
                        <p class="split-info">Split between: ${splitUsers}</p>
                    </div>
                    <div class="expense-amount">
                        <p class="amount">$${expense.amount}</p>
                        ${isOwner ? `
                            <div class="expense-actions">
                                <button class="btn btn-sm btn-secondary" onclick="openEditExpense('${expense.id}')">Edit</button>
                                <button class="btn btn-sm btn-danger delete-btn" onclick="deleteExpense('${expense.id}')">Delete</button>
                            </div>
                        ` : ''}
                    </div>
                </div>
            `;
        }).join('');

        expensesContainer.innerHTML = expenseCards;
    } catch (error) {
        showToast('Error loading expenses: ' + error.message, 'error');
    }
}

async function calculateUserBalance(tripId, userId) {
    try {
        // Get all expenses for this trip
        const { data: expenses, error: expensesError } = await supabase
            .from('expenses')
            .select('*, expense_splits(*)')
            .eq('trip_id', tripId);

        if (expensesError) throw expensesError;

        let balance = 0;

        expenses.forEach(expense => {
            // If user paid for this expense
            if (expense.paid_by === userId) {
                balance += parseFloat(expense.amount);
            }

            // Subtract what user owes
            const userSplit = expense.expense_splits.find(split => split.user_id === userId);
            if (userSplit) {
                balance -= parseFloat(userSplit.amount);
            }
        });

        return balance;
    } catch (error) {
        console.error('Error calculating balance:', error);
        return 0;
    }
}

async function calculateAndShowBalances() {
    try {
        const balances = {};

        // Initialize balances for all participants
        participants.forEach(p => {
            balances[p.id] = { name: p.name, balance: 0 };
        });

        // Get all expenses
        const { data: expenses, error } = await supabase
            .from('expenses')
            .select('*, expense_splits(*)')
            .eq('trip_id', currentTrip);

        if (error) throw error;

        // Calculate balances
        expenses.forEach(expense => {
            const payerId = expense.paid_by;
            
            // Add to payer's balance
            if (balances[payerId]) {
                balances[payerId].balance += parseFloat(expense.amount);
            }

            // Subtract from each person who owes
            expense.expense_splits.forEach(split => {
                if (balances[split.user_id]) {
                    balances[split.user_id].balance -= parseFloat(split.amount);
                }
            });
        });

        // Display balances
        const balanceItems = Object.entries(balances)
            .filter(([id, data]) => Math.abs(data.balance) > 0.01)
            .sort((a, b) => Math.abs(b[1].balance) - Math.abs(a[1].balance)) // Sort by amount
            .map(([id, data]) => {
                const amount = data.balance;
                const isPositive = amount > 0;
                const displayAmount = Math.abs(amount).toFixed(2);
                const text = isPositive ? 
                    `${data.name} gets back` :
                    `${data.name} owes`;

                return `
                    <div class="balance-item">
                        <span class="balance-name">${text}</span>
                        <span class="amount ${isPositive ? 'positive' : 'negative'}">
                            ${isPositive ? '+' : '-'}$${displayAmount}
                        </span>
                    </div>
                `;
            }).join('');

        balancesContainer.innerHTML = balanceItems || '<p class="empty-state">Nothing owed</p>';
    } catch (error) {
        showToast('Error calculating balances: ' + error.message, 'error');
    }
}

async function addExpense(description, amount, date, splitUserIds) {
    showLoading();
    try {
        // Create expense
        const { data: expense, error: expenseError } = await supabase
            .from('expenses')
            .insert({
                trip_id: currentTrip,
                paid_by: currentUser.id,
                description: description,
                amount: amount,
                expense_date: date
            })
            .select()
            .single();

        if (expenseError) throw expenseError;

        // Calculate split amount
        const splitAmount = (amount / splitUserIds.length).toFixed(2);

        // Create expense splits
        const splits = splitUserIds.map(userId => ({
            expense_id: expense.id,
            user_id: userId,
            amount: splitAmount
        }));

        const { error: splitsError } = await supabase
            .from('expense_splits')
            .insert(splits);

        if (splitsError) throw splitsError;

        closeModal('add-expense-modal');
        showToast('Expense added successfully!', 'success');
        await loadExpenses();
        await calculateAndShowBalances();
    } catch (error) {
        showToast('Error adding expense: ' + error.message, 'error');
    }
    hideLoading();
}

async function deleteExpense(expenseId) {
    if (!confirm('Are you sure you want to delete this expense?')) return;

    showLoading();
    try {
        const { error } = await supabase
            .from('expenses')
            .delete()
            .eq('id', expenseId);

        if (error) throw error;

        showToast('Expense deleted successfully!', 'success');
        await loadExpenses();
        await calculateAndShowBalances();
    } catch (error) {
        showToast('Error deleting expense: ' + error.message, 'error');
    }
    hideLoading();
}

window.deleteExpense = deleteExpense; // Make it globally accessible

async function openEditExpense(expenseId) {
    // Find the expense element and get its data
    const expenseElement = document.querySelector(`[data-expense*='"id":"${expenseId}"']`);
    if (!expenseElement) return;
    
    const expenseData = JSON.parse(expenseElement.getAttribute('data-expense'));
    
    // Populate the edit form
    document.getElementById('edit-expense-id').value = expenseData.id;
    document.getElementById('edit-expense-description').value = expenseData.description;
    document.getElementById('edit-expense-amount').value = expenseData.amount;
    document.getElementById('edit-expense-date').value = expenseData.expense_date;
    
    // Set up participant checkboxes
    const container = document.getElementById('edit-expense-participants');
    const checkboxes = participants.map(p => {
        const isChecked = expenseData.splits.includes(p.id);
        return `
            <label class="participant-checkbox">
                <input type="checkbox" value="${p.id}" ${isChecked ? 'checked' : ''}>
                ${p.name}
            </label>
        `;
    }).join('');
    
    container.innerHTML = checkboxes;
    
    // Show the modal
    showModal('edit-expense-modal');
}

window.openEditExpense = openEditExpense; // Make it globally accessible

async function updateExpense(expenseId, description, amount, date, splitUserIds) {
    showLoading();
    try {
        // Update the expense
        const { error: updateError } = await supabase
            .from('expenses')
            .update({
                description: description,
                amount: amount,
                expense_date: date
            })
            .eq('id', expenseId);

        if (updateError) throw updateError;

        // Delete existing splits
        const { error: deleteError } = await supabase
            .from('expense_splits')
            .delete()
            .eq('expense_id', expenseId);

        if (deleteError) throw deleteError;

        // Create new expense splits
        const splitAmount = (amount / splitUserIds.length).toFixed(2);
        const splits = splitUserIds.map(userId => ({
            expense_id: expenseId,
            user_id: userId,
            amount: splitAmount
        }));

        const { error: splitsError } = await supabase
            .from('expense_splits')
            .insert(splits);

        if (splitsError) throw splitsError;

        closeModal('edit-expense-modal');
        showToast('Expense updated successfully!', 'success');
        await loadExpenses();
        await calculateAndShowBalances();
    } catch (error) {
        showToast('Error updating expense: ' + error.message, 'error');
    }
    hideLoading();
}

async function removeParticipant(participantId, participantName) {
    if (!confirm(`Are you sure you want to remove ${participantName} from this trip? They will lose access to all trip information.`)) return;
    
    showLoading();
    try {
        // First check if this participant has any expenses
        const { data: expenses, error: expenseError } = await supabase
            .from('expenses')
            .select('id')
            .eq('trip_id', currentTrip)
            .eq('paid_by', participantId);
            
        if (expenseError) throw expenseError;
        
        if (expenses && expenses.length > 0) {
            showToast('Cannot remove participant who has added expenses. Please delete their expenses first.', 'error');
            hideLoading();
            return;
        }
        
        // Remove from expense splits
        const { error: splitError } = await supabase
            .from('expense_splits')
            .delete()
            .eq('user_id', participantId)
            .in('expense_id', (
                await supabase
                    .from('expenses')
                    .select('id')
                    .eq('trip_id', currentTrip)
            ).data.map(e => e.id));
            
        if (splitError) throw splitError;
        
        // Remove from trip participants
        const { error } = await supabase
            .from('trip_participants')
            .delete()
            .eq('trip_id', currentTrip)
            .eq('user_id', participantId);
            
        if (error) throw error;
        
        showToast('Participant removed successfully', 'success');
        
        // Reload participants and update UI
        await loadParticipants();
        showParticipantsList();
        
        // Reload expenses and balances since splits may have changed
        await loadExpenses();
        await calculateAndShowBalances();
        
    } catch (error) {
        showToast('Error removing participant: ' + error.message, 'error');
    }
    hideLoading();
}

window.removeParticipant = removeParticipant; // Make it globally accessible

async function deleteTrip() {
    if (!confirm('Are you sure you want to delete this trip? This will delete all expenses and cannot be undone.')) return;

    showLoading();
    try {
        const { error } = await supabase
            .from('trips')
            .delete()
            .eq('id', currentTrip);

        if (error) throw error;

        showToast('Trip deleted successfully!', 'success');
        showTripList();
    } catch (error) {
        showToast('Error deleting trip: ' + error.message, 'error');
    }
    hideLoading();
}

function generateInviteLink() {
    return `${window.location.origin}/splitwiz?trip=${currentTrip}`;
}

async function copyInviteLink() {
    const inviteLink = generateInviteLink();
    
    try {
        await navigator.clipboard.writeText(inviteLink);
        const btn = document.getElementById('copy-invite-btn');
        const originalText = btn.textContent;
        btn.textContent = 'Copied!';
        setTimeout(() => {
            btn.textContent = originalText;
        }, 2000);
        showToast('Invite link copied to clipboard!', 'success');
    } catch (err) {
        // If clipboard fails, select the text in the input field for manual copying
        const inviteLinkInput = document.getElementById('invite-link');
        inviteLinkInput.select();
        inviteLinkInput.setSelectionRange(0, 99999); // For mobile devices
        showToast('Clipboard not available. Please copy the selected link manually.', 'info');
    }
}

function showParticipantsList() {
    const container = document.getElementById('participants-list-container');
    const isOwner = window.currentTripData && window.currentTripData.created_by === currentUser.id;
    
    const participantItems = participants.map(p => {
        const canRemove = isOwner && p.id !== currentUser.id; // Owner can remove others but not themselves
        
        return `
            <div class="participant-item">
                <div class="participant-info">
                    <span class="participant-name">${p.name}</span>
                    <span class="participant-status">${p.email}</span>
                </div>
                ${canRemove ? `
                    <button class="btn btn-sm btn-danger" onclick="removeParticipant('${p.id}', '${p.name}')">
                        Remove
                    </button>
                ` : ''}
            </div>
        `;
    }).join('');
    
    container.innerHTML = participantItems || '<p class="empty-state">No participants yet</p>';
    
    // Set the invite link
    const inviteLinkInput = document.getElementById('invite-link');
    inviteLinkInput.value = generateInviteLink();
}

function showExpenseParticipants() {
    const container = document.getElementById('expense-participants');
    const checkboxes = participants.map(p => `
        <label class="participant-checkbox">
            <input type="checkbox" value="${p.id}" checked>
            ${p.name}
        </label>
    `).join('');
    
    container.innerHTML = checkboxes;
}

// Event Listeners
document.getElementById('google-login-btn').addEventListener('click', signInWithGoogle);
document.getElementById('logout-btn').addEventListener('click', signOut);
document.getElementById('new-trip-btn').addEventListener('click', () => showModal('new-trip-modal'));
document.getElementById('back-to-trips').addEventListener('click', showTripList);
document.getElementById('add-expense-btn').addEventListener('click', () => {
    showExpenseParticipants();
    // Set default date to today
    document.getElementById('expense-date').value = new Date().toISOString().split('T')[0];
    showModal('add-expense-modal');
});
document.getElementById('manage-participants-btn').addEventListener('click', () => {
    showParticipantsList();
    showModal('manage-participants-modal');
});
document.getElementById('delete-trip-btn').addEventListener('click', deleteTrip);

document.getElementById('new-trip-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const name = document.getElementById('trip-name-input').value;
    await createTrip(name);
    document.getElementById('trip-name-input').value = '';
});

document.getElementById('add-expense-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const description = document.getElementById('expense-description').value;
    const amount = parseFloat(document.getElementById('expense-amount').value);
    const date = document.getElementById('expense-date').value;
    
    const checkboxes = document.querySelectorAll('#expense-participants input[type="checkbox"]:checked');
    const splitUserIds = Array.from(checkboxes).map(cb => cb.value);
    
    if (splitUserIds.length === 0) {
        showToast('Please select at least one person to split with', 'error');
        return;
    }
    
    await addExpense(description, amount, date, splitUserIds);
    document.getElementById('expense-description').value = '';
    document.getElementById('expense-amount').value = '';
    document.getElementById('expense-date').value = new Date().toISOString().split('T')[0];
});

document.getElementById('copy-invite-btn').addEventListener('click', copyInviteLink);

document.getElementById('edit-expense-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const expenseId = document.getElementById('edit-expense-id').value;
    const description = document.getElementById('edit-expense-description').value;
    const amount = parseFloat(document.getElementById('edit-expense-amount').value);
    const date = document.getElementById('edit-expense-date').value;
    
    const checkboxes = document.querySelectorAll('#edit-expense-participants input[type="checkbox"]:checked');
    const splitUserIds = Array.from(checkboxes).map(cb => cb.value);
    
    if (splitUserIds.length === 0) {
        showToast('Please select at least one person to split with', 'error');
        return;
    }
    
    await updateExpense(expenseId, description, amount, date, splitUserIds);
});

// Initialize app
async function initApp() {
    // Check current session
    const { data: { session }, error } = await supabase.auth.getSession();
    
    if (session) {
        currentUser = session.user;
        userEmail.textContent = currentUser.email;
        // Ensure smooth transition
        setTimeout(() => {
            loginScreen.classList.remove('active');
            mainApp.classList.add('active');
            
            // Check if user came from an invite link
            const urlParams = new URLSearchParams(window.location.search);
            const inviteTripId = urlParams.get('trip');
            
            if (inviteTripId) {
                handleInviteLink(inviteTripId);
            } else {
                showTripList();
            }
        }, 100);
    } else {
        // Ensure login screen is properly displayed
        setTimeout(() => {
            loginScreen.classList.add('active');
            mainApp.classList.remove('active');
        }, 100);
    }
}

async function handleInviteLink(tripId) {
    showLoading();
    try {
        // Check if already a participant
        const { data: existing } = await supabase
            .from('trip_participants')
            .select('id')
            .eq('trip_id', tripId)
            .eq('user_id', currentUser.id)
            .single();
        
        if (!existing) {
            // Add as participant
            const { error } = await supabase
                .from('trip_participants')
                .insert({
                    trip_id: tripId,
                    user_id: currentUser.id
                });
            
            if (error) {
                console.error('Error joining trip:', error);
                showToast('Error joining trip. The link may be invalid.', 'error');
                showTripList();
                return;
            }
            
            showToast('Successfully joined the trip!', 'success');
        }
        
        // Clear URL params and show the trip
        window.history.replaceState({}, document.title, window.location.pathname);
        showTripDetail(tripId);
    } catch (error) {
        console.error('Error handling invite:', error);
        showTripList();
    }
    hideLoading();
}

// Auth state change listener
supabase.auth.onAuthStateChange((event, session) => {
    if ((event === 'SIGNED_IN' || event === 'INITIAL_SESSION') && session) {
        currentUser = session.user;
        userEmail.textContent = currentUser.email;
        // Small delay to prevent visual glitch during transition
        setTimeout(() => {
            loginScreen.classList.remove('active');
            mainApp.classList.add('active');
            showTripList();
        }, 100);
    } else if (event === 'SIGNED_OUT') {
        currentUser = null;
        // Small delay to prevent visual glitch during transition
        setTimeout(() => {
            loginScreen.classList.add('active');
            mainApp.classList.remove('active');
        }, 100);
    }
});

// Start the app
initApp();