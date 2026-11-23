document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('predictionForm');
    const tipBox = document.getElementById('tipBox');

    // only the predict page has the form
    if (!form) {
        return;
    }

    const studyInput = document.getElementById('study_hours');
    const socialInput = document.getElementById('social_hours');
    const sleepInput = document.getElementById('sleep_hours');
    const exerciseInput = document.getElementById('exercise_freq');

    form.addEventListener('submit', function(e) {
        const study = parseFloat(studyInput.value);
        const social = parseFloat(socialInput.value);
        const sleep = parseFloat(sleepInput.value);

        let warnings = [];
        if (study && study > 12) {
            warnings.push('Study hours are more than 12 a day. Is that right?');
        }
        if (social && social > 5 && study && study < 3) {
            warnings.push('High social media and low study hours.');
        }
        if (sleep && (sleep < 5 || sleep > 10)) {
            warnings.push('Sleep is outside 5-10 hours.');
        }

        if (warnings.length > 0) {
            if (!confirm(warnings.join('\n\n') + '\n\nContinue anyway?')) {
                e.preventDefault();
            }
        }
    });

    function updateTip() {
        const study = parseFloat(studyInput.value) || 0;
        const social = parseFloat(socialInput.value) || 0;
        const sleep = parseFloat(sleepInput.value) || 0;
        const exercise = parseFloat(exerciseInput.value) || 0;

        let message = 'Fill in your habits to see a tip.';
        let type = 'info';

        if (study >= 5 && sleep >= 7 && social < 2 && exercise >= 3) {
            message = 'Great habits, this should give a good score.';
            type = 'success';
        } else if (study >= 4 && sleep >= 7) {
            message = 'Good study routine, keep it up.';
            type = 'success';
        } else if (study < 2) {
            message = 'Study hours are low. Try 4-5 hours a day.';
            type = 'warning';
        } else if (social > 5) {
            message = 'Too much social media. Try to keep it under 2 hours.';
            type = 'warning';
        } else if (sleep < 6) {
            message = 'Not enough sleep. 7-9 hours is best.';
            type = 'warning';
        } else if (sleep > 9) {
            message = 'Sleeping a lot. 7-9 hours is usually enough.';
            type = 'warning';
        } else if (exercise < 2) {
            message = 'Try adding some exercise to your week.';
            type = 'info';
        }

        tipBox.className = 'alert alert-' + type;
        tipBox.innerHTML = '<strong>Tip</strong><p class="mb-0">' + message + '</p>';
    }

    studyInput.addEventListener('input', updateTip);
    socialInput.addEventListener('input', updateTip);
    sleepInput.addEventListener('input', updateTip);
    exerciseInput.addEventListener('input', updateTip);
});
