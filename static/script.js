    document.addEventListener('DOMContentLoaded', function() {
        
        // Form validation for predict page
        const predictionForm = document.getElementById('predictionForm');
        if (predictionForm) {
            predictionForm.addEventListener('submit', function(e) {
                const studyHours = parseFloat(document.querySelector('[name="study_hours"]')?.value);
                const socialHours = parseFloat(document.querySelector('[name="social_hours"]')?.value);
                const sleepHours = parseFloat(document.querySelector('[name="sleep_hours"]')?.value);
                
                const warnings = [];
                
                if (studyHours && studyHours > 12) {
                    warnings.push('Study hours exceed 12 hours per day. Please verify your input.');
                }
                
                if (socialHours && socialHours > 5 && studyHours && studyHours < 3) {
                    warnings.push('High social media usage combined with low study hours may impact academic performance.');
                }
                
                if (sleepHours && (sleepHours < 5 || sleepHours > 10)) {
                    warnings.push('Sleep outside 5-10 hours range may affect academic performance.');
                }
                
                if (warnings.length > 0) {
                    const confirmMessage = warnings.join('\n\n') + '\n\nDo you want to continue with these values?';
                    if (!confirm(confirmMessage)) {
                        e.preventDefault();
                    }
                }
            });
        }
        
        // Real-time feedback on predict page
        const studyInput = document.querySelector('[name="study_hours"]');
        const socialInput = document.querySelector('[name="social_hours"]');
        const sleepInput = document.querySelector('[name="sleep_hours"]');
        const exerciseInput = document.querySelector('[name="exercise_freq"]');
        const tipBox = document.getElementById('tipBox');
        
        function updateTip() {
            if (!tipBox) return;
            
            const study = parseFloat(studyInput?.value) || 0;
            const social = parseFloat(socialInput?.value) || 0;
            const sleep = parseFloat(sleepInput?.value) || 0;
            const exercise = parseFloat(exerciseInput?.value) || 0;
            
            let message = '';
            let icon = 'lightbulb';
            let alertClass = 'alert-info';
            
            if (study >= 5 && sleep >= 7 && social < 2 && exercise >= 3) {
                message = 'Optimal study habits detected. Your current routine suggests strong academic potential.';
                icon = 'trophy';
                alertClass = 'alert-success';
            } else if (study >= 4 && sleep >= 7) {
                message = 'Good study routine. Maintaining these habits can lead to consistent academic performance.';
                icon = 'check-circle';
                alertClass = 'alert-success';
            } else if (study < 2) {
                message = 'Study hours below recommended range. Consider increasing to 4-5 hours daily for better results.';
                icon = 'exclamation-triangle';
                alertClass = 'alert-warning';
            } else if (social > 5) {
                message = 'High social media usage detected. Limiting to under 2 hours daily may improve focus.';
                icon = 'mobile-alt';
                alertClass = 'alert-warning';
            } else if (sleep < 6) {
                message = 'Insufficient sleep detected. Adults need 7-9 hours for optimal cognitive function.';
                icon = 'moon';
                alertClass = 'alert-warning';
            } else if (sleep > 9) {
                message = 'Excessive sleep detected. While rest is important, 7-9 hours is generally optimal.';
                icon = 'moon';
                alertClass = 'alert-warning';
            } else if (exercise < 2) {
                message = 'Regular exercise boosts brain function. Consider adding physical activity to your routine.';
                icon = 'heartbeat';
                alertClass = 'alert-info';
            } else {
                message = 'Fill in your daily habits to receive personalized recommendations for academic improvement.';
                icon = 'lightbulb';
                alertClass = 'alert-info';
            }
            
            tipBox.className = `alert ${alertClass} mb-4`;
            tipBox.innerHTML = `
                <div class="d-flex align-items-start">
                    <i class="fas fa-${icon} fa-lg me-3" style="margin-top: 2px;"></i>
                    <div>
                        <strong>Insight</strong>
                        <p class="mb-0">${message}</p>
                    </div>
                </div>
            `;
        }
        
        studyInput?.addEventListener('input', updateTip);
        socialInput?.addEventListener('input', updateTip);
        sleepInput?.addEventListener('input', updateTip);
        exerciseInput?.addEventListener('input', updateTip);
        
        // Initialize tooltips
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.forEach(function(tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    });