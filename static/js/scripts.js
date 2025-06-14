// JavaScript for dashboard interactivity

document.addEventListener('DOMContentLoaded', () => {
  const tabs = document.querySelectorAll('.tabs button');

  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');

      // Additional logic for tabs can go here
      // Example: show/hide content sections based on tab
    });
  });

  // You can add dynamic updates (e.g., fetch trend via API)
  fetch('/api/accuracy')
    .then(res => res.json())
    .then(data => console.log('Accuracy:', data));

  fetch('/api/trend')
    .then(res => res.json())
    .then(data => console.log('Trend:', data));
});
