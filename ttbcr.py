import requests
import re
import math
import random
from flask import Flask, jsonify
from collections import defaultdict

app = Flask(__name__)

# =====================================================================
# BIẾN TOÀN CỤC - LƯU TRỮ LỊCH SỬ DỰ ĐOÁN ĐỂ ĐÁNH GIÁ HIỆU SUẤT MÔ HÌNH
# Tách biệt bộ nhớ dự đoán theo từng bàn chơi tránh xung đột dữ liệu giữa các bàn
# =====================================================================
global_model_predictions = defaultdict(lambda: {
    'trend': {},
    'short': {},
    'mean': {},
    'switch': {},
    'bridge': {},
    'asymmetric': {},
    'bias': {}
})

class BaccaratHybridEngineAI:
    def __init__(self, table_name):
        self.table_name = table_name
        self.raw_history = []       # Dữ liệu gốc bao gồm P, B, T (Con/Cái/Hòa)
        self.engine_history = []    # Định dạng chuẩn hóa: [{'session': x, 'result': 'Con'/'Cái', 'score': y, 'after_tie': True/False}]
        self.tie_positions = []     # Lưu trữ chỉ số các phiên xảy ra Hòa để phân tích hành vi sau Hòa
        self.markov_model = defaultdict(lambda: {'Con': 0, 'Cái': 0})
        self.markov_order = 3

    def load_and_sync_data(self, result_string):
        """Đồng bộ chuỗi kết quả sàn, gán mã phiên và lọc tách biệt Hòa nhưng không xóa bỏ tín hiệu"""
        if not result_string:
            self.engine_history = []
            return
        
        self.raw_history = list(result_string)
        self.engine_history = []
        self.tie_positions = []
        
        session_counter = 1001
        is_next_after_tie = False
        
        for idx, char in enumerate(self.raw_history):
            if char == 'T':
                self.tie_positions.append(idx)
                is_next_after_tie = True
                continue
                
            if char in ['P', 'B']:
                res_mapped = 'Con' if char == 'P' else 'Cái'
                simulated_score = 11 if char == 'P' else 7
                
                self.engine_history.append({
                    'session': f"SESS_{session_counter}",
                    'result': res_mapped,
                    'score': simulated_score,
                    'after_tie': is_next_after_tie
                })
                session_counter += 1
                is_next_after_tie = False
                
        # Học và nạp dữ liệu liên tục dựa trên toàn bộ lịch sử hiện tại của bàn này
        self._train_markov_layer(self.engine_history)

    def _train_markov_layer(self, history_slice):
        """Cập nhật ma trận học máy liên tục cho thuật toán cầu chuỗi lặp nâng cao theo từng bàn riêng biệt"""
        self.markov_model.clear()
        if len(history_slice) < self.markov_order + 1:
            return
        for i in range(len(history_slice) - self.markov_order):
            state = tuple(history_slice[j]['result'] for j in range(i, i + self.markov_order))
            next_val = history_slice[i + self.markov_order]['result']
            self.markov_model[state][next_val] += 1

    def detect_streak_and_break(self, history_slice):
        """Phát hiện độ dài chuỗi bệt hiện tại và tính toán xác suất bẻ gãy dựa trên lát cắt lịch sử cụ thể"""
        if not history_slice or len(history_slice) == 0:
            return {'streak': 0, 'currentResult': None, 'breakProb': 0}

        streak = 1
        current_result = history_slice[-1]['result']
        
        for i in range(len(history_slice) - 2, -1, -1):
            if history_slice[i]['result'] == current_result:
                streak += 1
            else:
                break

        last_15 = [item['result'] for item in history_slice[-15:]]
        if not last_15:
            return {'streak': streak, 'currentResult': current_result, 'breakProb': 0}

        switches = 0
        for i in range(1, len(last_15)):
            if last_15[i] != last_15[i-1]:
                switches += 1

        con_count = last_15.count('Con')
        cai_count = last_15.count('Cái')
        imbalance = abs(con_count - cai_count) / len(last_15)

        break_prob = 0
        if streak >= 8:
            break_prob = min(0.6 + switches / 15 + imbalance * 0.15, 0.9)
        elif streak >= 5:
            break_prob = min(0.35 + switches / 10 + imbalance * 0.25, 0.85)
        elif streak >= 3 and switches >= 7:
            break_prob = 0.3

        return {'streak': streak, 'currentResult': current_result, 'breakProb': break_prob}

    def evaluate_model_performance(self, current_history, model_name, lookback=10):
        """Đánh giá hiệu suất dự đoán thực tế của từng thuật toán thành phần dựa trên lát cắt lịch sử cụ thể"""
        preds = global_model_predictions[self.table_name][model_name]
        if not preds or len(current_history) < 2:
            return 1.0
        
        lookback = min(lookback, len(current_history) - 1)
        correct_predictions = 0

        for i in range(lookback):
            prev_session_obj = current_history[len(current_history) - (i + 2)]
            session_id = prev_session_obj['session']
            prediction = preds.get(session_id, 0)
            
            actual_result = current_history[len(current_history) - (i + 1)]['result']
            
            if (prediction == 1 and actual_result == 'Con') or (prediction == 2 and actual_result == 'Cái'):
                correct_predictions += 1

        performance_ratio = 1 + (correct_predictions - lookback / 2) / (lookback / 2) if lookback > 0 else 1.0
        return max(0.5, min(1.5, performance_ratio))

    def analyze_bias_and_trends(self, history_slice):
        """Thuật toán nhận diện Cầu Nghiêng vĩ mô bám xu hướng tránh bẻ ngược dòng"""
        if len(history_slice) < 15:
            return 0 
            
        last_20 = [item['result'] for item in history_slice[-20:]]
        con_count = last_20.count('Con')
        cai_count = last_20.count('Cái')
        
        if con_count >= 13: 
            return 1 
        elif cai_count >= 13: 
            return 2
        return 0

    def detect_asymmetric_patterns(self, history_slice):
        """Thuật toán quét cấu trúc cầu nhảy tuần hoàn phức tạp (1-2, 1-3, 2-3)"""
        if len(history_slice) < 9:
            return 0
            
        last_9 = [item['result'] for item in history_slice[-9:]]
        str_9 = "".join(['P' if x == 'Con' else 'B' for x in last_9])
        
        if str_9[-6:] in ["PBBPBB", "BPPBPP"]: 
            return 1 if str_9[-1] == 'B' else 2
        if str_9[-8:] in ["PBBBPBBB", "BPPPBPPP"]: 
            return 1 if str_9[-1] == 'B' else 2
        if str_9[-5:] in ["PPBBB", "BBPPP"]: 
            return 1 if str_9[-1] == 'B' else 2
            
        return 0

    def analyze_tie_influence(self, history_slice):
        """Thuật toán khai thác hành vi của dòng chảy ngay sau tiếng Hòa (Tie)"""
        if len(history_slice) < 5 or not history_slice[-1]['after_tie']:
            return 0 
            
        after_tie_con = 0
        after_tie_cai = 0
        for item in history_slice[:-1]:
            if item['after_tie']:
                if item['result'] == 'Con':
                    after_tie_con += 1
                else:
                    after_tie_cai += 1
                    
        if after_tie_con > after_tie_cai:
            return 1
        elif after_tie_cai > after_tie_con:
            return 2
        return 0

    def smart_bridge_break(self, history_slice):
        if len(history_slice) < 3:
            return {'prediction': 0, 'breakProb': 0, 'reason': 'Không đủ dữ liệu để bẻ cầu'}

        streak_data = self.detect_streak_and_break(history_slice)
        streak = streak_data['streak']
        current_result = streak_data['currentResult']
        break_prob = streak_data['breakProb']

        last_20_results = [item['result'] for item in history_slice[-20:]]
        last_20_scores = [item['score'] for item in history_slice[-20:]]

        final_break_prob = break_prob
        reason = ''
        
        avg_score = sum(last_20_scores) / (len(last_20_scores) or 1)
        score_deviation = sum(abs(score - avg_score) for score in last_20_scores) / (len(last_20_scores) or 1)
        
        last_5_results = last_20_results[-5:]
        pattern_counts = {}

        for i in range(len(last_20_results) - 2):
            pattern = ",".join(last_20_results[i:i+3])
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

        most_common_pattern = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)
        has_repeating_pattern = len(most_common_pattern) > 0 and most_common_pattern[0][1] >= 3

        if streak >= 6:
            final_break_prob = min(final_break_prob + 0.15, 0.9)
            reason = f"Chuỗi {streak} {current_result} dài, khả năng bẻ cầu cao"
        elif streak >= 4 and score_deviation > 3:
            final_break_prob = min(final_break_prob + 0.1, 0.85)
            reason = f"Biến động điểm lớn ({score_deviation:.1f}), khả năng bẻ tăng"
        elif has_repeating_pattern and all(res == current_result for res in last_5_results):
            final_break_prob = min(final_break_prob + 0.05, 0.8)
            reason = f"Phát hiện mẫu lặp {most_common_pattern[0][0]}, có khả năng bẻ cầu"
        else:
            final_break_prob = max(final_break_prob - 0.15, 0.15)
            reason = 'Thống kê an toàn, tiếp tục bám theo trục dọc xu hướng'

        prediction = (2 if current_result == 'Con' else 1) if final_break_prob > 0.65 else (1 if current_result == 'Con' else 2)
        return {'prediction': prediction, 'breakProb': final_break_prob, 'reason': reason}

    def trend_and_prob(self, history_slice):
        if len(history_slice) < 3: return 0
        
        streak_data = self.detect_streak_and_break(history_slice)
        streak = streak_data['streak']
        current_result = streak_data['currentResult']
        break_prob = streak_data['breakProb']
        
        if streak >= 5:
            if break_prob > 0.75: return 2 if current_result == 'Con' else 1
            return 1 if current_result == 'Con' else 2

        last_15_results = [item['result'] for item in history_slice[-15:]]
        if not last_15_results: return 0

        weighted_results = [math.pow(1.2, index) for index in range(len(last_15_results))]
        con_weight = sum(weighted_results[i] for i in range(len(last_15_results)) if last_15_results[i] == 'Con')
        cai_weight = sum(weighted_results[i] for i in range(len(last_15_results)) if last_15_results[i] == 'Cái')
        total_weight = con_weight + cai_weight

        last_10 = last_15_results[-10:]
        patterns = []
        if len(last_10) >= 4:
            for i in range(len(last_10) - 3):
                patterns.append(",".join(last_10[i:i+4]))

        pattern_counts = {}
        for p in patterns:
            pattern_counts[p] = pattern_counts.get(p, 0) + 1

        most_common = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)
        if len(most_common) > 0 and most_common[0][1] >= 3:
            pattern_parts = most_common[0][0].split(',')
            return 1 if pattern_parts[-1] != last_10[-1] else 2

        if total_weight > 0 and (abs(con_weight - cai_weight) / total_weight) >= 0.25:
            return 2 if con_weight > cai_weight else 1

        return 1 if last_15_results[-1] == 'Cái' else 2

    def short_pattern(self, history_slice):
        if len(history_slice) < 3: return 0
        
        streak_data = self.detect_streak_and_break(history_slice)
        streak = streak_data['streak']
        current_result = streak_data['currentResult']
        break_prob = streak_data['breakProb']
        
        if streak >= 4:
            if break_prob > 0.75: return 2 if current_result == 'Con' else 1
            return 1 if current_result == 'Con' else 2

        last_8_results = [item['result'] for item in history_slice[-8:]]
        if not last_8_results: return 0

        patterns = []
        if len(last_8_results) >= 3:
            for i in range(len(last_8_results) - 2):
                patterns.append(",".join(last_8_results[i:i+3]))

        pattern_counts = {}
        for p in patterns:
            pattern_counts[p] = pattern_counts.get(p, 0) + 1

        most_common = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)
        if len(most_common) > 0 and most_common[0][1] >= 2:
            pattern_parts = most_common[0][0].split(',')
            return 1 if pattern_parts[-1] != last_8_results[-1] else 2

        return 1 if last_8_results[-1] == 'Cái' else 2

    def mean_deviation(self, history_slice):
        if len(history_slice) < 3: return 0
        
        streak_data = self.detect_streak_and_break(history_slice)
        streak = streak_data['streak']
        current_result = streak_data['currentResult']
        break_prob = streak_data['breakProb']
        
        if streak >= 4:
            if break_prob > 0.75: return 2 if current_result == 'Con' else 1
            return 1 if current_result == 'Con' else 2

        last_12_results = [item['result'] for item in history_slice[-12:]]
        if not last_12_results: return 0

        con_count = last_12_results.count('Con')
        cai_count = len(last_12_results) - con_count
        imbalance = abs(con_count - cai_count) / len(last_12_results)

        if imbalance < 0.35:
            return 1 if last_12_results[-1] == 'Cái' else 2

        return 1 if cai_count > con_count else 2

    def recent_switch(self, history_slice):
        if len(history_slice) < 3: return 0
        
        streak_data = self.detect_streak_and_break(history_slice)
        streak = streak_data['streak']
        current_result = streak_data['currentResult']
        break_prob = streak_data['breakProb']
        
        if streak >= 4:
            if break_prob > 0.75: return 2 if current_result == 'Con' else 1
            return 1 if current_result == 'Con' else 2

        last_10_results = [item['result'] for item in history_slice[-10:]]
        if not last_10_results: return 0

        switches = 0
        for i in range(1, len(last_10_results)):
            if last_10_results[i] != last_10_results[i-1]:
                switches += 1

        return 1 if last_10_results[-1] == 'Cái' else 2

    def is_bad_pattern(self, history_slice):
        if len(history_slice) < 3: return False

        last_15_results = [item['result'] for item in history_slice[-15:]]
        if not last_15_results: return False

        switches = 0
        for i in range(1, len(last_15_results)):
            if last_15_results[i] != last_15_results[i-1]:
                switches += 1

        streak_data = self.detect_streak_and_break(history_slice)
        return switches >= 9 or streak_data['streak'] >= 10

    def ai_htdd_logic(self, history_slice):
        if len(history_slice) < 3:
            return {'prediction': 'Con' if random.random() < 0.5 else 'Cái', 'reason': 'Dữ liệu ngắn, cân bằng động'}

        last_5_results = [item['result'] for item in history_slice[-5:]]
        last_5_scores = [item['score'] for item in history_slice[-5:]]
        con_count = last_5_results.count('Con')
        cai_count = last_5_results.count('Cái')

        last_3_results = [item['result'] for item in history_slice[-3:]]
        if ",".join(last_3_results) == 'Con,Cái,Con':
            return {'prediction': 'Cái', 'reason': 'Cầu nhảy đảo ngắn 1-1 (Con-Cái-Con)'}
        elif ",".join(last_3_results) == 'Cái,Con,Cái':
            return {'prediction': 'Con', 'reason': 'Cầu nhảy đảo ngắn 1-1 (Cái-Con-Cái)'}

        if len(history_slice) >= 4:
            last_4_results = [item['result'] for item in history_slice[-4:]]
            if ",".join(last_4_results) == 'Con,Con,Cái,Cái':
                return {'prediction': 'Con', 'reason': 'Nhịp đối xứng song hành 2-2 (2 Con-2 Cái)'}
            elif ",".join(last_4_results) == 'Cái,Cái,Con,Con':
                return {'prediction': 'Cái', 'reason': 'Nhịp đối xứng song hành 2-2 (2 Cái-2 Con)'}

        if len(history_slice) >= 9 and all(item['result'] == 'Con' for item in history_slice[-6:]):
            return {'prediction': 'Cái', 'reason': 'Chuỗi bệt cực hạn (6 tay Con) → Ép Cái'}
        elif len(history_slice) >= 9 and all(item['result'] == 'Cái' for item in history_slice[-6:]):
            return {'prediction': 'Con', 'reason': 'Chuỗi bệt cực hạn (6 tay Cái) → Ép Con'}

        avg_score = sum(last_5_scores) / (len(last_5_scores) or 1)
        if avg_score > 10:
            return {'prediction': 'Con', 'reason': f'Trọng tải trung bình cao ({avg_score:.1f}) → Theo Con'}
        elif avg_score < 8:
            return {'prediction': 'Cái', 'reason': f'Trọng tải trung bình thấp ({avg_score:.1f}) → Theo Cái'}

        if con_count > cai_count + 1:
            return {'prediction': 'Cái', 'reason': f'Mật độ Con lấn lướt ({con_count}/5) → Đánh Cái'}
        elif cai_count > con_count + 1:
            return {'prediction': 'Con', 'reason': f'Mật độ Cái lấn lướt ({cai_count}/5) → Đánh Con'}
        
        total_con = sum(1 for item in history_slice if item['result'] == 'Con')
        total_cai = sum(1 for item in history_slice if item['result'] == 'Cái')
        
        if total_con > total_cai + 2:
            return {'prediction': 'Cái', 'reason': 'Vĩ mô mất cân bằng, tổng Con vượt ngưỡng'}
        elif total_cai > total_con + 2:
            return {'prediction': 'Con', 'reason': 'Vĩ mô mất cân bằng, tổng Cái vượt ngưỡng'}
        
        return {'prediction': 'Con' if random.random() < 0.5 else 'Cái', 'reason': 'Cân bằng hệ thống động'}

    def execute_hybrid_slice_prediction(self, history_slice):
        """Thực thi tính toán và nạp bộ nhớ đệm dự đoán cho một lát cắt dữ liệu bất kỳ"""
        current_session = history_slice[-1]['session']
        table_preds = global_model_predictions[self.table_name]

        trend_pred = (2 if history_slice[-1]['result'] == 'Con' else 1) if len(history_slice) < 5 else self.trend_and_prob(history_slice)
        short_pred = (2 if history_slice[-1]['result'] == 'Con' else 1) if len(history_slice) < 5 else self.short_pattern(history_slice)
        mean_pred = (2 if history_slice[-1]['result'] == 'Con' else 1) if len(history_slice) < 5 else self.mean_deviation(history_slice)
        switch_pred = (2 if history_slice[-1]['result'] == 'Con' else 1) if len(history_slice) < 5 else self.recent_switch(history_slice)
        bridge_data = {'prediction': (2 if history_slice[-1]['result'] == 'Con' else 1), 'breakProb': 0, 'reason': 'Lịch sử ngắn'} if len(history_slice) < 5 else self.smart_bridge_break(history_slice)
        
        bias_pred = self.analyze_bias_and_trends(history_slice)
        asymmetric_pred = self.detect_asymmetric_patterns(history_slice)
        tie_pred = self.analyze_tie_influence(history_slice)
        
        ai_data = self.ai_htdd_logic(history_slice)

        table_preds['trend'][current_session] = trend_pred
        table_preds['short'][current_session] = short_pred
        table_preds['mean'][current_session] = mean_pred
        table_preds['switch'][current_session] = switch_pred
        table_preds['bridge'][current_session] = bridge_data['prediction']
        table_preds['bias'][current_session] = bias_pred if bias_pred > 0 else (1 if ai_data['prediction'] == 'Con' else 2)
        table_preds['asymmetric'][current_session] = asymmetric_pred if asymmetric_pred > 0 else (1 if ai_data['prediction'] == 'Con' else 2)

        perf_trend = self.evaluate_model_performance(history_slice, 'trend')
        perf_short = self.evaluate_model_performance(history_slice, 'short')
        perf_mean = self.evaluate_model_performance(history_slice, 'mean')
        perf_switch = self.evaluate_model_performance(history_slice, 'switch')
        perf_bridge = self.evaluate_model_performance(history_slice, 'bridge')
        perf_bias = self.evaluate_model_performance(history_slice, 'bias')
        perf_asymmetric = self.evaluate_model_performance(history_slice, 'asymmetric')

        w_trend = 0.15 * perf_trend
        w_short = 0.15 * perf_short
        w_mean = 0.15 * perf_mean
        w_switch = 0.15 * perf_switch
        w_bridge = 0.10 * perf_bridge
        w_bias = 0.15 * perf_bias                
        w_asymmetric = 0.10 * perf_asymmetric    
        w_aihtdd = 0.15

        con_score = 0.0
        cai_score = 0.0

        if trend_pred == 1: con_score += w_trend
        elif trend_pred == 2: cai_score += w_trend
        if short_pred == 1: con_score += w_short
        elif short_pred == 2: cai_score += w_short
        if mean_pred == 1: con_score += w_mean
        elif mean_pred == 2: cai_score += w_mean
        if switch_pred == 1: con_score += w_switch
        elif switch_pred == 2: cai_score += w_switch
        if bridge_data['prediction'] == 1: con_score += w_bridge
        elif bridge_data['prediction'] == 2: cai_score += w_bridge
        
        if bias_pred == 1: con_score += w_bias
        elif bias_pred == 2: cai_score += w_bias
        if asymmetric_pred == 1: con_score += w_asymmetric
        elif asymmetric_pred == 2: cai_score += w_asymmetric
        if tie_pred == 1: con_score += 0.12 
        elif tie_pred == 2: cai_score += 0.12

        if ai_data['prediction'] == 'Con': con_score += w_aihtdd
        else: cai_score += w_aihtdd

        if self.is_bad_pattern(history_slice):
            con_score *= 0.8
            cai_score *= 0.8

        last_10_res = [item['result'] for item in history_slice[-10:]]
        last_10_con = last_10_res.count('Con')
        if last_10_con >= 7:
            cai_score += 0.15
        elif last_10_con <= 3:
            con_score += 0.15

        if bridge_data['breakProb'] > 0.65:
            if bridge_data['prediction'] == 1: con_score += 0.2
            else: cai_score += 0.2

        if con_score > cai_score:
            final_pred_text = "Con"
            calculated_rate = min(98, int((con_score / (con_score + cai_score if con_score + cai_score > 0 else 1)) * 100))
        else:
            final_pred_text = "Cái"
            calculated_rate = min(98, int((cai_score / (con_score + cai_score if con_score + cai_score > 0 else 1)) * 100))

        if calculated_rate < 50: 
            calculated_rate = 50 + (calculated_rate % 45)

        addons_reason = []
        if bias_pred > 0: addons_reason.append("[Nghiêng Vĩ Mô]")
        if asymmetric_pred > 0: addons_reason.append("[Cầu Nhảy]")
        if tie_pred > 0: addons_reason.append("[Tín Hiệu Hòa]")
        
        str_addons = " ".join(addons_reason)
        final_reason = f"{ai_data['reason']} | {bridge_data['reason']}"
        if str_addons:
            final_reason = f"{str_addons} {final_reason}"
            
        return final_pred_text, f"{calculated_rate}%", final_reason

    def execute_flexible_hybrid_prediction(self):
        """Buộc hệ thống phân tích ngược 5 phiên cũ trước khi chốt dự đoán cho phiên hiện tại"""
        h = self.engine_history
        total_len = len(h)
        
        for offset in range(5, 0, -1):
            past_slice = h[:total_len - offset]
            self._train_markov_layer(past_slice)
            self.execute_hybrid_slice_prediction(past_slice)

        self._train_markov_layer(h)
        return self.execute_hybrid_slice_prediction(h)

def parse_and_sort_tables(item):
    name = item.get("table_name", "")
    match = re.match(r"([a-zA-Z]*)\s*(\d+)", str(name))
    if match:
        prefix, num = match.groups()
        return (0 if prefix == "" else 1, prefix, int(num))
    return (2, str(name), 0)

@app.route('/')
def hybrid_baccarat_dashboard():
    api_url = "https://bcr-thanhnhatx.onrender.com/data"
    
    try:
        response = requests.get(api_url, timeout=7)
        if response.status_code != 200:
            return jsonify({"status": "error", "message": f"Lỗi phản hồi hệ thống API: {response.status_code}"}), 500
        raw_data = response.json()
    except Exception as e:
        return jsonify({"status": "error", "message": f"Mất kết nối máy chủ dữ liệu API gốc. Chi tiết: {e}"}), 500

    sorted_tables = sorted(raw_data, key=parse_and_sort_tables)
    final_view_data = []

    for item in sorted_tables:
        table_name = item.get("table_name", "Unknown")
        result_string = item.get("result", "")
        time_stamp = item.get("time", "--:--:--")

        engine = BaccaratHybridEngineAI(table_name=table_name)
        engine.load_and_sync_data(result_string)
        
        total_sessions = len(engine.engine_history)
        full_chain_clean = "".join(['P' if x['result'] == 'Con' else 'B' for x in engine.engine_history])

        # Phải nạp đủ tối thiểu 15 phiên (10 phiên nền móng + 5 phiên học/phân tích ngược)
        if total_sessions < 15:
            final_view_data.append({
                "Bàn": table_name,
                "Chuỗi": result_string,
                "Phiên kế tiếp": f"SESS_{1001 + total_sessions}",
                "Dự đoán": "Đang nạp dữ liệu nền móng...",
                "Tỉ lệ": f"{total_sessions}/15 phiên"
            })
            continue

        prediction, rate, analysis_reason = engine.execute_flexible_hybrid_prediction()
        next_session_id = f"SESS_{1001 + total_sessions}"

        # Sửa cấu trúc thông báo và key trả về đồng nhất theo định dạng yêu cầu
        final_view_data.append({
            "Bàn": table_name,
            "Chuỗi": result_string,
            "Phiên kế tiếp": next_session_id,
            "Dự đoán": prediction,
            "Tỉ lệ": rate,
            "Phân tích hệ thống": analysis_reason
        })

    return jsonify({
        "status": "success",
        "total_tables": len(final_view_data),
        "results": final_view_data
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=True)